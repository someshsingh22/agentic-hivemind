"""Hypothesis -> row features and a frozen regression specification, via a pi agent."""

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .features import validate_features
from .models import agent, pi_home
from .prompts import render_prompt

SCHEMA = Path(__file__).with_name("schema.json")


def decompose(i, hypothesis, env, cfg, out, home, columns):
    """Reuse the hypothesis JSON if present, as on main; otherwise generate it."""
    path = Path(out, "decomposed", f"{i}.json").resolve()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        system, task = render_prompt(
            "hypothesis_decomposer",
            csv=Path(env["csv"]).resolve(), description=env.get("description", ""),
            columns=", ".join(columns), target=env["target"], hypothesis=hypothesis,
        )
        agent(task, system, home, cfg, Path(home, "schema.json"), path)
    result = json.loads(path.read_text())
    try:
        validate_features(result["features"], columns)
    except (KeyError, ValueError) as error:
        raise ValueError(f"{path}: invalid feature contract: {error}") from error
    return {"id": i, "hypothesis": hypothesis, **result}


def decompose_all(hypotheses, env, cfg, out):
    with open(env["csv"], newline="") as source:
        columns = next(csv.reader(source))
    forbidden = {env["target"], env.get("group"), "post_id", "comment_id"}
    columns = [column for column in columns if column not in forbidden]
    schema = json.loads(SCHEMA.read_text())
    if cfg.get("constructs", False):
        constructs = {
            "type": "object",
            "properties": {
                "surface_features": {"type": "array", "items": {"type": "string"}},
                "theoretical_construct": {"type": ["string", "null"]},
                "secondary_constructs": {"type": "array", "maxItems": 5,
                                         "items": {"type": "string"}},
                "domain": {"type": "string"},
            },
            "required": ["surface_features", "theoretical_construct", "secondary_constructs", "domain"],
            "additionalProperties": False,
        }
        schema["properties"] = {"constructs": constructs, **schema["properties"]}
        schema["required"] = ["constructs", *schema["required"]]
    schema["properties"]["features"]["items"]["properties"]["source_columns"]["items"]["enum"] = columns
    home = pi_home(cfg, Path(out, "pi"))
    (home / "schema.json").write_text(json.dumps(schema, indent=2))
    with ThreadPoolExecutor(cfg["workers"]) as pool:
        return list(pool.map(
            lambda ih: decompose(*ih, env, cfg, out, home, columns), hypotheses.items(),
        ))
