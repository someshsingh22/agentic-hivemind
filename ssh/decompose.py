"""Hypothesis -> reusable variables (features) + relationship (H0/H1 and one regression test), via a pi agent."""

import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from os import fspath
from pathlib import Path

from .models import agent, pi_home
from .prompts import prompt_config, render_prompt

SCHEMA = Path(__file__).with_name("schema.json")


def _decompose(i, hypothesis, cfg, out, home, columns, resolved_prompt):
    path = Path(out, "decomposed", f"{i}.json").resolve()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        system, task = resolved_prompt
        agent(task, system, home, cfg, home / "schema.json", path)
    result = json.loads(path.read_text())
    for feature in result["features"]:
        sources = feature.get("source_columns")
        if (not isinstance(sources, list) or not sources
                or any(not isinstance(source, str) or source not in columns for source in sources)
                or len(set(sources)) != len(sources)):
            raise ValueError(f"{path}: {feature['name']} needs valid original source_columns; use a fresh run for old outputs")
    return {"id": i, "hypothesis": hypothesis, **result}


def _inputs(hypotheses, env, cfg):
    with open(env["csv"], newline="") as source:
        columns = next(csv.reader(source))
    forbidden = {env["target"], env.get("group"), "post_id", "comment_id"}
    columns = [column for column in columns if column not in forbidden]
    schema = json.loads(SCHEMA.read_text())
    schema["properties"]["features"]["items"]["properties"]["source_columns"]["items"]["enum"] = columns
    with open(env["csv"], "rb") as source:
        csv_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    resolved_prompts = []
    for i, hypothesis in hypotheses.items():
        system, user = render_prompt(
            "hypothesis_decomposer",
            csv=Path(env["csv"]).resolve(), description=env.get("description", ""),
            columns=", ".join(columns), target=env["target"], hypothesis=hypothesis,
        )
        resolved_prompts.append({"id": i, "system": system, "user": user})
    inputs = {
        "version": 1,
        "hypotheses": [{"id": i, "hypothesis": text} for i, text in hypotheses.items()],
        "env": env,
        "agent": cfg,
        "csv": {"path": str(Path(env["csv"]).resolve()), "sha256": csv_sha256},
        "schema": schema,
        "prompt": prompt_config("hypothesis_decomposer"),
        "resolved_prompts": resolved_prompts,
    }
    return json.loads(json.dumps(inputs, default=fspath)), columns


def _previous_inputs(out):
    manifest = Path(out, "decomposition_inputs.json")
    if manifest.exists():
        try:
            previous = json.loads(manifest.read_text())
            if not isinstance(previous, dict):
                raise ValueError("Input manifest must be an object")
            return previous
        except (ValueError, UnicodeError) as error:
            raise ValueError(f"{manifest}: invalid input manifest; use a fresh output directory") from error
    if Path(out, "decomposed.json").exists() or any(Path(out, "decomposed").glob("*.json")):
        raise ValueError(f"{manifest}: cached decomposition has no matching input manifest; use a fresh output directory")
    return None


def _bind_inputs(inputs, previous, out):
    manifest = Path(out, "decomposition_inputs.json")
    if previous is not None:
        if previous != inputs:
            raise ValueError(f"{manifest}: decomposition inputs changed; use a fresh output directory")
    else:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(inputs, indent=2))


def decompose(i, hypothesis, env, cfg, out, home, columns):
    """Decompose one claim, requiring the same input identity on every cached resume."""
    previous = _previous_inputs(out)
    if previous is None:
        hypotheses = {i: hypothesis}
    else:
        try:
            hypotheses = {item["id"]: item["hypothesis"] for item in previous["hypotheses"]}
        except (KeyError, TypeError) as error:
            raise ValueError("Invalid decomposition input identity; use a fresh output directory") from error
        if i not in hypotheses or hypotheses[i] != hypothesis:
            raise ValueError("Decomposition hypothesis changed; use a fresh output directory")
    inputs, original_columns = _inputs(hypotheses, env, cfg)
    if list(columns) != original_columns:
        raise ValueError("Decomposition predictor columns changed; use the original CSV predictor columns")
    _bind_inputs(inputs, previous, out)
    home = Path(home)
    (home / "schema.json").write_text(json.dumps(inputs["schema"], indent=2))
    prompt = next(prompt for prompt in inputs["resolved_prompts"] if prompt["id"] == i)
    return _decompose(i, hypothesis, cfg, out, home, columns, (prompt["system"], prompt["user"]))


def decompose_all(hypotheses, env, cfg, out):
    previous = _previous_inputs(out)
    inputs, columns = _inputs(hypotheses, env, cfg)
    _bind_inputs(inputs, previous, out)
    home = pi_home(cfg, Path(out, "pi"))
    (home / "schema.json").write_text(json.dumps(inputs["schema"], indent=2))
    prompts = {prompt["id"]: (prompt["system"], prompt["user"]) for prompt in inputs["resolved_prompts"]}
    with ThreadPoolExecutor(cfg["workers"]) as pool:
        return list(pool.map(
            lambda ih: _decompose(*ih, cfg, out, home, columns, prompts[ih[0]]),
            hypotheses.items(),
        ))
