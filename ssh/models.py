"""pi agent and embedding clients, configured entirely from config.yaml + env."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
from openai import OpenAI


def pi_home(cfg, home):
    """Write pi's models.json (provider + model from the config) into a run-local pi home."""
    home = Path(home).resolve()
    home.mkdir(parents=True, exist_ok=True)
    model = {"id": cfg["model"], "reasoning": True, "maxTokens": cfg["max_tokens"], "compat": cfg["compat"]}
    provider = {"baseUrl": os.path.expandvars(cfg["base_url"]), "api": cfg["api"], "apiKey": f"${cfg['api_key_env']}", "models": [model]}
    Path(home, "models.json").write_text(json.dumps({"providers": {cfg["provider"]: provider}}))
    return home


def agent(prompt, system, home, cfg):
    """Run one non-interactive pi session inside `home`."""
    subprocess.run(
        ["pi", "-p", "--session-dir", str(home / "sessions"), "--no-context-files", "--no-extensions", "--no-skills",
         "--provider", cfg["provider"], "--model", cfg["model"], "--thinking", cfg["thinking"],
         "--tools", cfg["tools"], "--append-system-prompt", system, prompt],
        cwd=home, env={**os.environ, "PI_CODING_AGENT_DIR": str(home)}, check=True, stdout=subprocess.DEVNULL,
    )


def embed(texts, cfg):
    client = OpenAI(base_url=os.path.expandvars(cfg["base_url"]), api_key=os.environ[cfg["api_key_env"]], max_retries=8)
    vectors = np.array([d.embedding for d in client.embeddings.create(model=cfg["model"], input=texts).data])
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
