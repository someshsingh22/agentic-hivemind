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
    """Merge features by source-aware cosine clustering -> OUT/plan.json."""
    from .dedup import dedup, summary
    from .calibration import write_calibration

    cfg, env, out = setup(env, config)
    decomposed = read(out / "decomposed.json")["decomposed"]
    plan = dedup(decomposed, env, cfg)
    write(out / "plan.json", plan)
    write_calibration(plan, out, cfg["dedup"]["threshold"])
    return summary(plan)


def test(env, csv, config="config.yaml"):
    """Run frozen tests on a CSV with one column per plan feature -> OUT/results.csv."""
    from .stats import run

    cfg, env, out = setup(env, config)
    results = run(read(out / "plan.json"), pd.read_csv(csv), cfg["test"]["alpha"])
    results.to_csv(out / "results.csv", index=False)
    return {"tested": int(results.p.notna().sum()), "ssh": int(results.ssh.sum()), "total": len(results)}


def tune(env, config="config.yaml"):
    """Decompose once and compare three cosine-distance thresholds; no annotation/tests."""
    from .calibration import write_calibration
    from .dedup import dedup, summary
    from .models import embed

    decompose(env, config)
    cfg, env, out = setup(env, config)
    decomposed = read(out / "decomposed.json")["decomposed"]
    features = [feature for claim in decomposed for feature in claim["features"]]
    vectors = embed([f"{f['name']}: {f['definition']}" for f in features], cfg["embedding"]) if len(features) > 1 else None
    results = {}
    for threshold in (0.20, 0.25, 0.30):
        trial = out / f"threshold_{threshold:.2f}"
        trial_cfg = {**cfg, "dedup": {**cfg["dedup"], "threshold": threshold}}
        print(f"Starting threshold {threshold:.2f}", flush=True)
        plan = dedup(decomposed, env, trial_cfg, vectors=vectors)
        write(trial / "config.json", trial_cfg)
        write(trial / "plan.json", plan)
        write_calibration(plan, trial, threshold)
        results[f"{threshold:.2f}"] = summary(plan)
        write(out / "tuning_summary.json", results)
        print(json.dumps({"threshold": threshold, **summary(plan)}), flush=True)
    return results


def compare_descriptions(decomposed, config="config.yaml"):
    """Compare embedding representations on saved features; never decompose or annotate."""
    from .calibration import write_calibration
    from .dedup import dedup, summary
    from .features import validate_features
    from .models import embed

    load_dotenv()
    cfg = yaml.safe_load(Path(config).read_text())
    source = Path(decomposed)
    data = read(source)
    env, claims = data["env"], data["decomposed"]
    for claim in claims:
        validate_features(claim["features"])
    features = [feature for claim in claims for feature in claim["features"]]
    texts = {
        "short_definition": [f"{f['name']}: {f['definition']}" for f in features],
        "full_description": [f"{f['name']}: {f['spec']} (dtype: {f['dtype']})" for f in features],
    }
    count = len(features)
    vectors = embed(texts["short_definition"] + texts["full_description"], cfg["embedding"]) if count > 1 else None
    out = source.parent / "description_comparison"
    write(out / "inputs.json", {
        "source": str(source.resolve()), "embedding": cfg["embedding"],
        "feature_order": [{"hypothesis": d["id"], "name": f["name"]} for d in claims for f in d["features"]],
        "texts": texts,
    })
    results = {}
    for index, representation in enumerate(texts):
        view_vectors = vectors[index * count:(index + 1) * count] if vectors is not None else None
        results[representation] = {}
        for threshold in (0.20, 0.25, 0.30):
            trial = out / representation / f"threshold_{threshold:.2f}"
            trial_cfg = {**cfg, "dedup": {**cfg["dedup"], "threshold": threshold}}
            print(f"Comparing {representation} at threshold {threshold:.2f}", flush=True)
            plan = dedup(claims, env, trial_cfg, vectors=view_vectors)
            write(trial / "config.json", trial_cfg)
            write(trial / "plan.json", plan)
            write_calibration(plan, trial, threshold)
            results[representation][f"{threshold:.2f}"] = summary(plan)
            write(out / "comparison_summary.json", results)
            print(json.dumps({"representation": representation, "threshold": threshold, **summary(plan)}), flush=True)
    return results


fire.Fire({"decompose": decompose, "dedup": dedup, "tune": tune,
           "compare_descriptions": compare_descriptions, "test": test})
