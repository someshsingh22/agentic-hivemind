# agentic-hivemind

Code for the Agentic Hivemind paper. `ssh/` turns a bank of hypotheses into deduplicated features and frozen
statistical tests. Once an annotated CSV has the feature columns, it reports **SSH** (statistically significant hypotheses).

```
define features & tests -> deduplicate measurements -> annotate rows (external) -> run tests -> report SSH
     ssh decompose               ssh dedup                                          ssh test
```

1. **decompose**: a [pi](https://www.npmjs.com/package/@earendil-works/pi-coding-agent) agent (Sonnet 5.5) per
   hypothesis. It inspects the dataset, then answers through a terminating `submit` tool (`ssh/submit.ts`) whose
   arguments must match `ssh/schema.json`. Strict tool decoding is on, and pi returns validation errors to the model.
   The answer has reusable *features* (`name, dtype, definition, kind=code|prompt, spec`) and a *relationship*:
   H0/H1, an OLS formula, the tested `term`, and the claimed `sign`.
   Sessions are saved in `runs/<env>/pi/sessions`.
2. **dedup**: embeds `name: definition` and runs complete-linkage clustering on cosine distance. Features from the same
   hypothesis, or with different dtypes, never merge. Formulas are rewritten to canonical names. Output: `plan.json`
   with the feature registry, which says what to annotate, and the frozen tests.
3. **test**: fits OLS with cluster-robust SEs by the env `group` (HC1 if there is none). It applies Holm correction
   across testable hypotheses. `ssh` means `p_holm <= alpha` and the coefficient sign matches the claim.

All parameters are in `config.yaml`: models, thinking effort, embedding model, threshold, alpha, seed, and sample size.
Datasets are in `envs/<env>.yaml`: target, group, CSV, hypotheses JSON, and a column description.
Secrets are read from env vars (see `.env.example`).

```bash
npm i -g @earendil-works/pi-coding-agent
cp .env.example .env                        # or: source /sensei-fs/users/someshs/secrets.sh
uv run python -m ssh decompose envs/cmv.yaml
uv run python -m ssh dedup envs/cmv.yaml
uv run python -m ssh test envs/cmv.yaml annotated.csv   # annotated.csv has one column per plan feature
```
