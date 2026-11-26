"""Render editable agent prompts from the ExperiGen-style YAML configuration."""

from pathlib import Path
from string import Template

import yaml

PROMPTS = Path(__file__).with_name("prompts.yaml")


def prompt_config(name):
    """Load one agent's templates and shared policy without caching edits."""
    document = yaml.safe_load(PROMPTS.read_text())
    config = document["agents"][name]["prompt_config"]
    policy = document["measurement_policy"]
    if not isinstance(policy, str) or not policy.strip():
        raise ValueError(f"{PROMPTS}: measurement_policy must be a nonempty string")
    for key in ("system_message", "user_template"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{PROMPTS}: agents.{name}.prompt_config.{key} must be a nonempty string")
    return {**{key: config[key] for key in ("system_message", "user_template")}, "measurement_policy": policy}


def render_prompt(name, **context) -> tuple[str, str]:
    """Return (system, user), expanding only the known policy marker in system."""
    config = prompt_config(name)
    system = config["system_message"].replace("$measurement_policy", config["measurement_policy"])
    return system, Template(config["user_template"]).substitute(context)
