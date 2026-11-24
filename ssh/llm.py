import json
import os

import numpy as np
from anthropic import Anthropic
from openai import OpenAI

BASE = f"https://{os.environ['AZURE_RESOURCE_NAME']}.services.ai.azure.com"
KEY = os.environ["AZURE_API_KEY"]
claude = Anthropic(base_url=f"{BASE}/anthropic", api_key=KEY, max_retries=8)
azure = OpenAI(base_url=f"{BASE}/openai/v1", api_key=KEY, max_retries=8)


def structured(system, user, schema, model="claude-sonnet-5-5"):
    """One Claude call whose output is constrained to a JSON schema."""
    msg = claude.messages.create(
        model=model,
        max_tokens=8000,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    return json.loads(next(b.text for b in msg.content if b.type == "text"))


def embed(texts, model="text-embedding-3-large"):
    vectors = np.array([d.embedding for d in azure.embeddings.create(model=model, input=texts).data])
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
