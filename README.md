# agentic-hivemind

Code for the Agentic Hivemind paper. `ssh/` turns hypotheses into scalar row-level feature extractors,
automatically deduplicates semantic features with Sonnet, and freezes their statistical tests.
Annotation and SSH testing are separate stages. The dedup evaluation has no human-label or pending-resolution step.

## Feature extraction

The feature contract follows the manager's [GitHub main branch](https://github.com/someshsingh22/agentic-hivemind/blob/main/ssh/decompose.py)
and [ExperiGen prod](https://github.com/someshsingh22/experigen/blob/prod/common_config.yaml).

- Each feature has exactly `name`, `dtype`, `source_columns`, `definition`, `kind` and `spec`.
- A feature returns one binary/numeric scalar or null per dataset row.
- A semantic prompt directly computes the requested measurement. "Any emotional comment" is one
  binary prompt feature, not a comment-label vector followed by mandatory aggregation code.
- Code computes deterministic measurements. Explicit numeric comparisons may retain a base measurement
  and add a code feature, as in manager main. Code features are not deduplicated.
- Sources are the original predictors actually read, including jointly required context. Outcomes,
  identifiers and generated feature names are not source columns.
- Each hypothesis retains H0/H1, its OLS formula, tested term, claimed sign and limitations. Hypotheses
  are not merged. The statistical evaluator is unchanged from manager main.

Prompts live in `ssh/prompts.yaml`. The decomposition prompt and shared measurement policy are unchanged
by the binary-review correction. Missing/malformed required input gives null; valid empty comment lists
give 0 for any/count and null for share/mean/all. Current supplied text includes edits unless the claim
restricts time. These conventions are operational definitions, not evidence of annotation accuracy.

The schema restricts sources to actual CSV predictors. The submit tool runs the Python contract validator
before saving decomposition output, returning contract errors to the agent. Structural validation does
not establish semantic correctness.

## Automated semantic deduplication

1. Embed `name: definition` for prompt features.
2. Select every pair within the cosine-distance cutoff whose original source-column sets and dtypes
   match and whose hypotheses differ. Source order is irrelevant.
3. Sonnet reviews each pair's complete extraction definitions and specs. It must return exactly one
   final verdict: `equivalent` or `distinct`, with a concrete reason. There is no alignment/abstention
   category or request for human resolution. Sonnet judges whether wording differences materially
   change the measurement; it cannot rewrite the definitions to make a merge possible.
4. Only equivalent pairs can merge. Final complete linkage requires every pair in a cluster to be
   within the cutoff, eligible and approved. Close A–B and B–C do not justify merging a rejected A–C.
5. Preserve canonical features, original members, verdicts/reasons and rewritten formula/code references
   in the plan. Every code feature remains separate.

Malformed or incomplete model output remains an error; it is not converted into a scientific verdict.
The evaluator is automated, but Sonnet's judgments remain fallible. Prompt/threshold tuning should be
reported as evaluator development, not independent evidence that a lower feature count is correct.

## Run the CMV trials

Inputs are `data/cmv.csv` and `data/hypotheses_ssh_eval/cmv.json`, configured in `envs/cmv.yaml`.
Set `ANTHROPIC_FOUNDRY_RESOURCE=mdsr-resource-dev-resource` and `ANTHROPIC_FOUNDRY_API_KEY` in `.env`.
Sonnet and `text-embedding-3-large` use that resource key. Never commit credentials.
Use Node 24 LTS and install pi with `npm i -g @earendil-works/pi-coding-agent`.

```bash
git pull --ff-only origin namit-rows
bash run_calibration.sh --tune
```

The launcher uses the existing `/mnt/localssd/namit/experigen-evals/.venv`, starts a nohup worker and tails
its log. Do not launch concurrent copies. It stops after deduplication, without Qwen annotation or SSH tests.
Set `VENV` to another existing environment if needed. Optional positional arguments select env/config YAML.

The three trials use cutoffs 0.20, 0.25 and 0.30 with the same decomposition and embeddings. Embeddings are
requested once per invocation, including resume. Pair verdicts are cached and reused across cutoffs and
identical resumes. Every eligible within-cutoff pair requires a review, so calls can exceed earlier
cluster-only runs. The three cutoffs are not independent model repetitions.

## Outputs and reuse of v5 features

Decomposition remains in `runs/calibration_v5/cmv/`, bound to its exact inputs by `decomposition_inputs.json`.
The binary evaluator writes fresh results under `runs/calibration_v5/cmv/automated/`:

- `dedup_reviews/`: fresh binary reviews, fingerprinted by complete feature inputs, prompt, schema and model.
- `threshold_0.20/`, `threshold_0.25/`, `threshold_0.30/`: each trial's effective config, plan and pair exports.
- `tuning_summary.json`: completed trial counts, including reviewed/equivalent/distinct pairs.

Existing v5 decompositions are reusable because their prompt, schema and inputs have not changed.
Old review verdicts and plans remain outside `automated/`; they are not relabeled or mixed into the new
protocol. If other decomposition inputs change, use a fresh output directory. Original experiment results
are not silently migrated. Return shared decomposition/input identity plus the `automated/` outputs for analysis.

Without `--tune`, the launcher runs a single configured cutoff, writing `automated/plan.json` and CSVs.
The explicit `ssh test` command reads that single-run plan; tuning itself never runs statistical tests.

## Pair audit exports

Each trial writes `feature_pairs.csv` for all semantic pairs, with original rubrics, sources, distance,
eligibility, Sonnet verdict/reason and merge status. Far/ineligible pairs can have blank verdicts; every
eligible within-cutoff pair must have a final binary decision. `same_candidate_cluster` is diagnostic only,
not a review gate.

`calibration_sample.csv` contains up to 10 eligible pairs per distance bin, sampled with seed 42, for
optional inspection. It requires no manual labels. `threshold_sensitivity.csv` reports candidate and
reviewed/equivalent/distinct/unreviewed counts across cutoffs. These are automated evaluator decisions,
not ground-truth precision/recall or an automatic recommendation of a best threshold.

The former human-label scoring command and pending-review categories have been removed.

## Local verification

```bash
uv run python -m unittest discover -s tests -v
```

Deterministic tests cover binary decisions, source/dtype/owner gates, all eligible close-pair coverage,
non-transitive approval, code/formula references, cache identity, resume and automated exports. A Node 24+
test executes the actual submit extension. A real nohup/tail launcher smoke exercises the workflow with
model/embedding boundaries simulated. No paid model calls run locally; these checks do not establish
Sonnet's semantic accuracy.
