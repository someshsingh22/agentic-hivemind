"""Review small candidate clusters, never dataset rows, before sharing annotations."""

import hashlib
import json
import os
from itertools import combinations
from pathlib import Path

from .models import agent, pi_home
from .prompts import render_prompt

SCHEMA = Path(__file__).with_name("equivalence_schema.json")


def validate_comparisons(comparisons, count):
    """Require one explicit decision for every unordered pair; fail closed on malformed reviews."""
    if not isinstance(comparisons, list):
        raise ValueError("Semantic review must contain a comparisons list")
    expected = set(combinations(range(count), 2))
    decisions = {}
    for item in comparisons:
        if not isinstance(item, dict) or set(item) != {"left", "right", "verdict", "reason"}:
            raise ValueError("Each semantic comparison needs left, right, verdict and reason")
        left, right = item["left"], item["right"]
        if type(left) is not int or type(right) is not int or not (0 <= left < count and 0 <= right < count):
            raise ValueError("Semantic comparison references an unavailable feature")
        pair = tuple(sorted((left, right)))
        if pair not in expected or pair in decisions:
            raise ValueError("Semantic review contains a duplicate or self comparison")
        if item["verdict"] not in ("equivalent", "distinct"):
            raise ValueError("Semantic comparison has an unknown verdict")
        if not isinstance(item["reason"], str) or not item["reason"].strip():
            raise ValueError("Semantic comparison needs a nonempty reason")
        decisions[pair] = {**item, "left": pair[0], "right": pair[1]}
    if set(decisions) != expected:
        raise ValueError("Semantic review is incomplete; every candidate pair needs a decision")
    return [decisions[pair] for pair in sorted(decisions)]


class EquivalenceReviewer:
    """Use Sonnet's submit-only session; cache by full rubrics, prompts, schema and model settings."""

    def __init__(self, cfg, out):
        self.cfg = {**cfg["agent"], "tools": "submit"}
        self.out = Path(out)

    def __call__(self, features):
        system, prompt = render_prompt("feature_equivalence", features=json.dumps(features, ensure_ascii=False, indent=2))
        schema = json.loads(SCHEMA.read_text())
        size = len(features)
        schema["properties"]["comparisons"].update(minItems=size * (size - 1) // 2, maxItems=size * (size - 1) // 2)
        for name in ("left", "right"):
            schema["properties"]["comparisons"]["items"]["properties"][name]["maximum"] = size - 1
        inputs = {
            "features": features,
            "system_message": system,
            "user_message": prompt,
            "agent": {**self.cfg, "base_url": os.path.expandvars(self.cfg["base_url"])},
            "schema": schema,
        }
        identity = hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        directory = self.out / "dedup_reviews"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{identity}.json"
        stamp = directory / f"{identity}.inputs.json"
        if path.exists():
            if not stamp.exists() or json.loads(stamp.read_text()) != inputs:
                raise ValueError(f"Semantic review inputs changed: {path}")
        else:
            stamp.write_text(json.dumps(inputs, ensure_ascii=False, indent=2))
            schema_path = directory / f"{identity}.schema.json"
            schema_path.write_text(json.dumps(schema, indent=2))
            home = pi_home(self.cfg, self.out / "pi_review")
            print(f"Reviewing semantic candidate cluster: {size} features, {size * (size - 1) // 2} pairs", flush=True)
            agent(prompt, system, home, self.cfg, schema_path, path)
        result = json.loads(path.read_text())
        if not isinstance(result, dict) or set(result) != {"comparisons"}:
            raise ValueError(f"Invalid semantic review object: {path}")
        return validate_comparisons(result["comparisons"], size)
