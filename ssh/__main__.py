"""python -m ssh {decompose,dedup,test} ENV.yaml: define features & tests -> dedup -> (annotate elsewhere) -> SSH."""

import json
import random
from pathlib import Path

import fire
import pandas as pd
import yaml
from dotenv import load_dotenv


def setup(env, config):
    load_dotenv()
    cfg, env = yaml.safe_load(Path(config).read_text()), yaml.safe_load(Path(env).read_text())
    return cfg, env, Path(cfg["out"], env["name"])


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))


def decompose(env, config="config.yaml"):
    """pi agent decomposes each hypothesis into features + a test -> OUT/decomposed.json."""
    from .decompose import decompose_all

    cfg, env, out = setup(env, config)
    hypotheses = dict(enumerate(read(env["hypotheses"])["hypotheses"]))
    if cfg["n_hypotheses"]:
        hypotheses = {i: hypotheses[i] for i in sorted(random.Random(cfg["seed"]).sample(list(hypotheses), cfg["n_hypotheses"]))}
    write(out / "decomposed.json", {"env": env, "decomposed": decompose_all(hypotheses, env, cfg["agent"], out)})


def dedup(env, config="config.yaml"):
    """Merge equivalent features -> OUT/plan.json (feature registry + frozen tests)."""
    from .dedup import dedup, summary
    from .equivalence import EquivalenceReviewer

    cfg, env, out = setup(env, config)
    plan = dedup(read(out / "decomposed.json")["decomposed"], env, cfg, EquivalenceReviewer(cfg, out))
    write(out / "plan.json", plan)
    return summary(plan)


def test(env, csv, config="config.yaml"):
    """Run frozen tests on a CSV with one column per plan feature -> OUT/results.csv."""
    from .stats import run

    cfg, env, out = setup(env, config)
    results = run(read(out / "plan.json"), pd.read_csv(csv), cfg["test"]["alpha"])
    results.to_csv(out / "results.csv", index=False)
    return {"tested": int(results.p.notna().sum()), "ssh": int(results.ssh.sum()), "total": len(results)}


fire.Fire({"decompose": decompose, "dedup": dedup, "test": test})
