# agentic-hivemind

Code for the Agentic Hivemind paper. `ssh/` turns a bank of hypotheses into deduplicated features and frozen
statistical tests. Once an annotated CSV has the feature columns, it reports **SSH** (statistically significant hypotheses).

```
define features & tests -> deduplicate measurements -> annotate rows (external) -> run tests -> report SSH
     ssh decompose               ssh dedup                                          ssh test
```

1. **decompose**: Claude (`claude-sonnet-5-5`, structured output) splits each hypothesis into reusable *features*
   (`name, dtype, definition, kind=code|prompt, spec`) and a *relationship*: H0/H1 plus an OLS formula,
   tested `term`, and claimed `sign`.
2. **dedup**: embeds `name: definition` (`text-embedding-3-large`) and runs complete-linkage clustering on cosine
   distance (`--threshold`, default 0.25). Features from the same hypothesis, or with different dtypes, never merge.
   Formulas are rewritten to canonical feature names. Output: `plan.json` with the feature registry, which says
   what to annotate, and the frozen tests.
3. **test**: fits OLS with cluster-robust SEs by the env `group` (HC1 if there is none). It applies Holm correction
   across testable hypotheses. `ssh` means `p_holm <= alpha` and the coefficient sign matches the claim.

```bash
source /sensei-fs/users/someshs/secrets.sh   # AZURE_API_KEY, AZURE_RESOURCE_NAME
uv run python -m ssh decompose envs/cmv.yaml hypotheses.json runs/cmv --n 10
uv run python -m ssh dedup runs/cmv
uv run python -m ssh test runs/cmv annotated.csv   # annotated.csv has one column per plan feature
```

Hypotheses JSON: `{"hypotheses": ["...", ...]}`. Env YAML: `name, target, group, description`
(the description documents columns for the decomposer).
