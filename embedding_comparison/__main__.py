"""Compare embedding inputs without changing frozen measurements or merge logic."""

import json
from pathlib import Path
from string import Template

import fire
import numpy as np
import yaml
from dotenv import load_dotenv

from ssh.calibration import write_calibration
from ssh.dedup import dedup, summary
from ssh.features import validate_features
from ssh.models import agent, embed, pi_home

PACKAGE = Path(__file__).resolve().parent


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def run(decomposed, config="config.yaml", out="runs/embedding_inputs/cmv"):
    """Compare name:definition and Sonnet-compressed specs; no annotation or tests."""
    load_dotenv()
    source = Path(decomposed).resolve()
    bank = json.loads(source.read_text())
    cfg = yaml.safe_load(Path(config).read_text())
    claims, env = bank["decomposed"], bank["env"]
    for claim in claims:
        validate_features(claim["features"])
    entries = [(claim["id"], feature) for claim in claims
               for feature in claim["features"] if feature["kind"] == "prompt"]
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name, value in (("decomposed.json", bank), ("config.json", cfg)):
        if not (out / name).exists():
            write_json(out / name, value)
    prompt_path = out / "prompts.yaml"
    if not prompt_path.exists():
        prompt_path.write_text((PACKAGE / "prompts.yaml").read_text())
    prompts = yaml.safe_load(prompt_path.read_text())["agents"]["describe"]["prompt_config"]
    agent_cfg = {**cfg["agent"], "tools": "submit"}
    home = None
    concise = []
    for index, (owner, feature) in enumerate(entries):
        path = out / "descriptions" / f"{index:03d}.json"
        if not path.exists():
            if home is None:
                home = pi_home(agent_cfg, out / "pi")
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Describing {index + 1}/{len(entries)}: H{owner} {feature['name']}", flush=True)
            prompt = Template(prompts["user_template"]).substitute(feature=json.dumps(feature, indent=2))
            agent(prompt, prompts["system_message"], home, agent_cfg, PACKAGE / "schema.json", path)
        response = json.loads(path.read_text())
        if set(response) != {"text"} or not isinstance(response["text"], str) or not response["text"].strip():
            raise ValueError(f"{path}: expected a nonempty text description")
        concise.append(response["text"])
    texts = {
        "short_definition": [f"{f['name']}: {f['definition']}" for _, f in entries],
        "concise_measurement": concise,
    }
    write_json(out / "inputs.json", {
        "source": str(source), "embedding": cfg["embedding"],
        "feature_order": [{"hypothesis": owner, "name": f["name"]} for owner, f in entries],
        "texts": texts,
    })
    results = {}
    for representation, inputs in texts.items():
        vector_path = out / representation / "vectors.json"
        if len(inputs) < 2:
            vectors = np.ones((len(inputs), 1))
        elif vector_path.exists():
            vectors = np.array(json.loads(vector_path.read_text()))
        else:
            vectors = embed(inputs, cfg["embedding"])
            write_json(vector_path, vectors.tolist())
        if (vectors.ndim != 2 or len(vectors) != len(inputs)
                or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1)):
            raise ValueError(f"{vector_path}: expected one normalized vector per semantic feature")
        results[representation] = {}
        for threshold in (0.20, 0.25, 0.30):
            trial = out / representation / f"threshold_{threshold:.2f}"
            trial_cfg = {**cfg, "dedup": {**cfg["dedup"], "threshold": threshold}}
            plan = dedup(claims, env, trial_cfg, vectors=vectors)
            write_json(trial / "config.json", trial_cfg)
            write_json(trial / "plan.json", plan)
            write_calibration(plan, trial, threshold)
            counts = summary(plan)
            results[representation][f"{threshold:.2f}"] = counts
            print(json.dumps({"representation": representation, "threshold": threshold, **counts}), flush=True)
        write_json(out / "comparison_summary.json", results)
    return results


if __name__ == "__main__":
    fire.Fire({"run": run})
