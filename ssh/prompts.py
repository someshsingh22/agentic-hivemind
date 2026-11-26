"""Render editable agent prompts from the ExperiGen-style YAML configuration."""

from pathlib import Path
from string import Template

import yaml

PROMPTS = Path(__file__).with_name("prompts.yaml")


def prompt_config(name):
    """Load one agent's system message and user template without caching edits."""
    config = yaml.safe_load(PROMPTS.read_text())["agents"][name]["prompt_config"]
    for key in ("system_message", "user_template"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{PROMPTS}: agents.{name}.prompt_config.{key} must be a nonempty string")
    return {key: config[key] for key in ("system_message", "user_template")}


def render_prompt(name, **context) -> tuple[str, str]:
    """Return (system, user), substituting $variables only in the user template."""
    config = prompt_config(name)
    return config["system_message"], Template(config["user_template"]).substitute(context)
