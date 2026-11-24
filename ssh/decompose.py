"""Hypothesis -> reusable variables (features) + relationship (H0/H1 and one regression test)."""

from concurrent.futures import ThreadPoolExecutor

from .llm import structured

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

SCHEMA = {
    "type": "object",
    "properties": {
        "features": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "dtype": {"enum": ["binary", "numeric"]},
                    "definition": {"type": "string"},
                    "kind": {"enum": ["code", "prompt"]},
                    "spec": {"type": "string"},
                },
                "required": ["name", "dtype", "definition", "kind", "spec"],
                "additionalProperties": False,
            },
        },
        "h0": {"type": "string"},
        "h1": {"type": "string"},
        "rhs": {"type": "string"},
        "term": {"type": ["string", "null"]},
        "sign": {"enum": ["+", "-", "two_sided"]},
        "note": {"type": "string"},
    },
    "required": ["features", "h0", "h1", "rhs", "term", "sign", "note"],
    "additionalProperties": False,
}


def decompose(hypothesis, env):
    user = f"DATASET\n{env['description']}\nOutcome column: {env['target']}\n\nHYPOTHESIS\n{hypothesis}"
    return {"hypothesis": hypothesis, **structured(SYSTEM, user, SCHEMA)}


def decompose_all(hypotheses, env, workers=8):
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda h: decompose(h, env), hypotheses))
