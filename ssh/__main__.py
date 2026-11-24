"""python -m ssh {decompose,dedup,test}: define features & tests -> dedup -> (annotate elsewhere) -> SSH."""

import json
from pathlib import Path

import fire
import pandas as pd
import yaml


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))


def decompose(env, hypotheses, out, n=None):
    """Decompose hypotheses (JSON {"hypotheses": [...]}) into features + tests -> out/decomposed.json."""
    from .decompose import decompose_all

    env = yaml.safe_load(Path(env).read_text())
    write(Path(out, "decomposed.json"), {"env": env, "decomposed": decompose_all(read(hypotheses)["hypotheses"][:n], env)})


def dedup(out, threshold=0.25):
    """Merge equivalent features (cosine-distance threshold) -> out/plan.json (feature registry + frozen tests)."""
    from .dedup import dedup, summary

    d = read(Path(out, "decomposed.json"))
    plan = dedup(d["decomposed"], d["env"], threshold)
    write(Path(out, "plan.json"), plan)
    return summary(plan)


def test(out, csv, alpha=0.05):
    """Run frozen tests on a CSV with one column per plan feature -> out/results.csv."""
    from .stats import run

    results = run(read(Path(out, "plan.json")), pd.read_csv(csv), alpha)
    results.to_csv(Path(out, "results.csv"), index=False)
    return {"tested": int(results.p.notna().sum()), "ssh": int(results.ssh.sum()), "total": len(results)}


fire.Fire({"decompose": decompose, "dedup": dedup, "test": test})
