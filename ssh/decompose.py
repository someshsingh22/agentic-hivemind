"""Hypothesis -> reusable variables (features) + relationship (H0/H1 and one regression test), via a pi agent."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .models import agent, pi_home

SYSTEM = """You decompose ONE scientific hypothesis about a dataset into (a) the variables it needs and
(b) the relationship it claims, expressed as a single regression test. Do not judge or rewrite the hypothesis.

FEATURES are reusable base measurements of one row, not claims. Many hypotheses will share them,
so make each feature atomic and generic enough to be reused (e.g. "op_concedes_weakness", not
"op_concedes_weakness_increases_delta"). Each feature has:
- name: unique snake_case identifier.
- dtype: binary (0/1) or numeric. Express categories as binary indicators.
- definition: one short canonical line "<text scope or column>: <exact quantity>", no boilerplate,
  e.g. "OP argument body: count of first-person singular pronouns". Used to match features across hypotheses.
- kind: "code" if computable deterministically from raw columns (counts, lengths, regex, thresholds,
  existing numeric columns), else "prompt" (needs an LLM/human annotator to read text).
- spec: for code, a precise pandas/python expression or recipe over raw columns or other features;
  for prompt, a precise annotation rubric: which column(s) and speaker scope to read, what counts,
  and the exact output (0/1, or the numeric scale with anchors).
Preserve every qualifier of the hypothesis (speaker, text scope, subset, unit). If the hypothesis
fixes a threshold or comparison (">= 100", "outnumber", "above median"), add a binary code feature
derived from the base feature, and keep the base feature too.
Never use the outcome or identifier columns as features.

RELATIONSHIP:
- h0 and h1: the null and alternative hypotheses in one sentence each, in terms of the feature names.
- rhs: right-hand side of a patsy/statsmodels OLS formula (outcome is added automatically) using
  ONLY feature names, e.g. "a", "a + control", "a * moderator", "a + I(a ** 2)".
- term: the exact design-matrix column whose coefficient tests the claim, e.g. "a", "a:moderator",
  "I(a ** 2)". Null if the claim cannot be tested by one coefficient.
- sign: "+" or "-" for the claimed direction of that coefficient on the outcome, "two_sided" if none.
- note: assumptions, proxies, or why it is untestable; empty string otherwise."""

TASK = """DATASET: {csv}
First inspect it (e.g. with python/pandas): the columns the hypothesis needs, their dtypes, value
ranges, and a few example values, so every feature spec is grounded in the actual data.
{description}
Outcome column: {target}

HYPOTHESIS: {hypothesis}

Write ONLY this JSON object to {path}:
{{"features": [{{"name": str, "dtype": "binary"|"numeric", "definition": str, "kind": "code"|"prompt", "spec": str}}],
 "h0": str, "h1": str, "rhs": str, "term": str|null, "sign": "+"|"-"|"two_sided", "note": str}}"""


def decompose(i, hypothesis, env, cfg, out, home):
    path = Path(out, "decomposed", f"{i}.json").resolve()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        task = TASK.format(csv=Path(env["csv"]).resolve(), description=env["description"], target=env["target"], hypothesis=hypothesis, path=path)
        agent(task, SYSTEM, home, cfg)
    return {"id": i, "hypothesis": hypothesis, **json.loads(path.read_text())}


def decompose_all(hypotheses, env, cfg, out):
    home = pi_home(cfg, Path(out, "pi"))
    with ThreadPoolExecutor(cfg["workers"]) as pool:
        return list(pool.map(lambda ih: decompose(*ih, env, cfg, out, home), hypotheses.items()))
