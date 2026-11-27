"""python -m sonnet_dedup run DECOMPOSED.json --config config.yaml --out runs/sonnet_only/cmv."""

import json
import os
import subprocess
from pathlib import Path
from string import Template

import fire
import yaml
from dotenv import load_dotenv

from ssh.dedup import summary
from ssh.models import pi_home

from .core import build_plan, prepare_groups, validate_partition

PACKAGE = Path(__file__).resolve().parent


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def run(decomposed, config="config.yaml", out="runs/sonnet_only/cmv"):
    """Partition saved semantic features with Sonnet; never decompose, embed or annotate."""
    load_dotenv()
    source = Path(decomposed).resolve()
    bank = json.loads(source.read_text())
    batches = prepare_groups(bank["decomposed"], bank["env"])
    agent_cfg = {**yaml.safe_load(Path(config).read_text())["agent"], "tools": "submit"}
    prompts = yaml.safe_load((PACKAGE / "prompts.yaml").read_text())
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for path, value in ((out / "decomposed.json", bank), (out / "config.json", {"agent": agent_cfg})):
        if not path.exists():
            write_json(path, value)
    if not (out / "prompts.yaml").exists():
        (out / "prompts.yaml").write_text((PACKAGE / "prompts.yaml").read_text())
    home = pi_home(agent_cfg, out / "pi")
    partitions = []
    for index, entries in enumerate(batches):
        prefix = out / "groups" / f"{index:03d}"
        input_path = prefix.with_suffix(".json")
        schema_path = prefix.with_suffix(".schema.json")
        response_path = prefix.with_suffix(".response.json")
        if response_path.exists():
            print(f"Reusing group {index + 1}/{len(batches)}: {len(entries)} semantic features", flush=True)
            response = json.loads(response_path.read_text())
        else:
            write_json(input_path, entries)
            schema = json.loads((PACKAGE / "schema.json").read_text())
            schema["properties"]["groups"]["items"]["properties"]["members"]["items"]["enum"] = [entry["key"] for entry in entries]
            write_json(schema_path, schema)
            if len({entry["hypothesis_id"] for entry in entries}) < 2:
                print(f"Keeping group {index + 1}/{len(batches)} separate: no cross-hypothesis pair", flush=True)
                response = {"groups": [{"members": [entry["key"]],
                                       "reason": "Not reviewed: no cross-hypothesis pair in this source/type group."}
                                      for entry in entries]}
                write_json(response_path, response)
            else:
                print(f"Reviewing group {index + 1}/{len(batches)}: {len(entries)} semantic features", flush=True)
                prompt = Template(prompts["user"]).substitute(features=json.dumps(entries, indent=2))
                subprocess.run(
                    ["pi", "-p", "--session-dir", str(home / "sessions"), "--no-context-files", "--no-extensions", "--no-skills",
                     "--provider", agent_cfg["provider"], "--model", agent_cfg["model"], "--thinking", agent_cfg["thinking"],
                     "-e", str(PACKAGE / "submit.ts"), "--tools", "submit", "--append-system-prompt", prompts["system"], prompt],
                    cwd=home, check=True, stdout=subprocess.DEVNULL,
                    env={**os.environ, "PI_CODING_AGENT_DIR": str(home), "SUBMIT_SCHEMA": str(schema_path),
                         "SUBMIT_OUT": str(response_path)},
                )
                response = json.loads(response_path.read_text())
        validate_partition(entries, response)
        partitions.extend(response["groups"])
    plan = build_plan(bank["decomposed"], bank["env"], partitions)
    write_json(out / "plan.json", plan)
    counts = summary(plan)
    write_json(out / "summary.json", counts)
    print(json.dumps(counts), flush=True)
    return counts


if __name__ == "__main__":
    fire.Fire({"run": run})
