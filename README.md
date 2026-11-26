# agentic-hivemind

Code for the Agentic Hivemind paper. `ssh/` decomposes hypotheses into row-level features and frozen
statistical tests, deduplicates semantic features, and evaluates an externally annotated CSV for
SSH (statistically significant hypotheses).

## Feature extraction

The feature contract follows the manager's [GitHub main branch](https://github.com/someshsingh22/agentic-hivemind/blob/main/ssh/decompose.py)
and [ExperiGen prod](https://github.com/someshsingh22/experigen/blob/prod/common_config.yaml):

- A feature returns one binary/numeric scalar, or null, per dataset row.
- Use a semantic prompt when the measurement requires language understanding. The prompt directly
  returns the requested quantity. For example, "any comment makes an emotional appeal" is one binary
  feature, not a comment-label vector followed by a separate code aggregation.
- Use code for deterministic measurements such as character counts, supplied numeric columns and
  arithmetic. As in manager main, an explicit threshold/comparison can retain its base measurement
  and add a code feature. This is not a mandatory split for semantic any/count/share measurements.
- Each feature has exactly `name`, `dtype`, `source_columns`, `definition`, `kind` and `spec`.
  `kind` is `prompt` or `code`. Sources are the original predictors actually read, including jointly
  required context; they exclude outcomes, identifiers and generated feature names.
- Each hypothesis retains H0/H1, its OLS formula, tested term, claimed sign and limitations.
  Hypotheses are not merged or discarded.

Prompts live in `ssh/prompts.yaml` under `agents.<name>.prompt_config`, using ExperiGen's YAML layout.
The shared measurement policy preserves source scope, attribution, qualifiers, missingness and units.
The schema restricts sources to the actual CSV predictors. The submit tool runs the Python contract
validator before saving output and returns errors to the agent for correction. These checks validate
structure and declared metadata, not the semantic accuracy of the generated rubric.

## Semantic-feature deduplication

`ssh dedup` embeds `name: definition` for prompt features only and reviews every eligible pair
within cosine distance `0.25`, one pair per Sonnet call. Source-column sets and dtypes must match;
same-hypothesis features cannot merge. Source order is ignored. Every code feature remains separate.
Candidate complete-linkage groups remain diagnostic only; they no longer exclude close cross-group pairs
from review. Final merges still use complete linkage with approval required for every pair.
Only `equivalent` permits merging. `distinct`, `uncertain` and `needs_alignment` stay separate.
Alignment flags do not rewrite measurements. Any/count/share, emotional/highly emotional and other
materially different measurements must remain distinct. Every pair in a final cluster must be approved.

`plan.json` contains canonical features, original members, rewritten code/formula references, frozen tests,
and review verdicts/reasons/distances. Reviews are cached by complete inputs, prompts, schema and model
settings. Review sessions use only `submit`; they do not annotate dataset rows. Model verdicts require
human inspection before annotation.

The statistical evaluator is unchanged from the manager's GitHub main branch: OLS, cluster-robust SEs by
the configured group (HC1 otherwise), Holm correction and the claimed coefficient sign.

## Run the CMV calibration

Inputs are already configured in `envs/cmv.yaml`:

- Dataset: `data/cmv.csv`.
- Hypotheses: `data/hypotheses_ssh_eval/cmv.json`, containing the supplied 100 hypotheses.

Set `ANTHROPIC_FOUNDRY_RESOURCE=mdsr-resource-dev-resource` and `ANTHROPIC_FOUNDRY_API_KEY` in `.env`.
Both Sonnet and `text-embedding-3-large` use that resource key. Never commit credentials.
Use Node 24 LTS and install pi with `npm i -g @earendil-works/pi-coding-agent`.

From the repository root on the experiment machine:

```bash
bash run_calibration.sh --tune
```

The launcher uses `/mnt/localssd/namit/experigen-evals/.venv`, starts a `nohup` worker and follows its
log with `tail`. Ctrl-C stops only the follower. Do not launch another copy while the worker is running.
It runs decomposition and semantic deduplication only, then stops. It does not call Qwen, annotate rows
or run SSH tests. Set `VENV` to use another existing environment; optional positional arguments select
the environment YAML and config YAML.

The default output is **`runs/calibration_v5/cmv/`**. This requires a fresh decomposition with the
consistent input/scope policy. Previous v1/v2/v3/v4 results remain untouched and are not silently migrated.
`decomposition_inputs.json` binds the CSV hash, exact hypotheses, environment, model settings, schema
and resolved prompts. An identical interrupted run reuses completed decompositions; changed inputs
require a fresh output directory.

Return the shared `decomposed.json`, `decomposition_inputs.json`, `dedup_reviews/`, `tuning_summary.json`
and all three trial directories for inspection. For a single calibration, return its root plan/CSVs instead.
Review session logs for credentials before publishing them.

## Three controlled tuning trials

`--tune` decomposes the 100 hypotheses once and embeds their semantic features once per invocation.
It then compares thresholds 0.20, 0.25 and 0.30 using the same features and vectors. Pair verdicts are
cached in a shared `dedup_reviews/`, so a pair already reviewed at a lower cutoff is not paid for again
at a higher cutoff or on an identical resumed run. Embeddings are requested again on restart.
Plans, effective configs and calibration CSVs live in `threshold_0.20/`, `threshold_0.25/` and
`threshold_0.30/`; `tuning_summary.json` contains completed trial counts and is updated after each trial.
An interrupted trial can resume through the same command. Decomposition identity checks still apply.

The prompt now explicitly assigns valid empty comment lists 0 for any/count and null for share/mean/all.
Missing/malformed required inputs remain null. It uses supplied current text including edits unless the
claim restricts time, and does not turn illustrative example lists into extra eligibility rules.
Incomplete OP context cannot establish a negative quotation match; an observed match against an available
permitted source may establish a positive. These are measurement conventions, not validated annotations.

Every eligible within-cutoff pair now requires a review. This can mean substantially more Sonnet calls
than v4's cluster-only review. Requests remain sequential; actual counts depend on the new decomposition.
The trials share verdicts but test different candidate cutoffs, not three independent model runs.
Do not select the trial with the fewest features automatically. Inspect new merges, alignment cases,
and human-labeled equivalents missed at each cutoff. Changing prompt and review coverage together means
a v4/v5 count difference cannot isolate either one's contribution. Within v5, the cutoff is the only
trial difference. No annotation or SSH tests run.

Without `--tune`, the launcher still runs one calibration at the configured threshold.


## Calibrate the distance threshold

Deduplication exports:

- `feature_pairs.csv`: every semantic pair, including ineligible and distant pairs, with original
  rubrics, sources, distances, eligibility, model verdicts and final merge state.
- `calibration_sample.csv`: up to 10 eligible pairs per distance bin, sampled deterministically with
  seed 42, including distances above 0.25. Human labels start blank.
- `threshold_sensitivity.csv`: candidate counts for thresholds 0.10 through 0.40.

Copy the sample to `reviewed_sample.csv` before labeling, so rerunning deduplication cannot overwrite
your work. Fill `review_label` with `equivalent`, `distinct` or `uncertain`, and optionally explain it in
`review_reason_human`. Model verdicts are not human labels. Then use the activated evaluation venv:

```bash
trial=runs/calibration_v5/cmv/threshold_0.25
cp "$trial/calibration_sample.csv" "$trial/reviewed_sample.csv"
# Edit reviewed_sample.csv before running the scoring command.
nohup python -u -m ssh calibrate \
  "$trial/reviewed_sample.csv" "$trial/human_calibration" \
  > "$trial/human_calibration.log" 2>&1 < /dev/null &
PID=$!
tail --pid="$PID" -n +1 -F "$trial/human_calibration.log"
```

Precision/recall describe only the explicitly human-reviewed eligible subset, not population accuracy
or final clustering accuracy. No threshold is automatically recommended or changed. Review approved
merges and missed equivalents rather than targeting a lower feature count.

## Local verification

```bash
uv run python -m unittest discover -s tests -v
```

Tests use deterministic/seeded embeddings and simulated model submissions. They cover source/dtype/owner
gates, direct row measurements, code/formula canonicalization, non-transitive pair decisions, cache
identity, complete pair exports and human-label calibration. A Node 24+ test executes the actual submit
extension and checks rejected submissions leave no output before a corrected submission succeeds.
A separate real nohup-launcher smoke exercised all three trials with shared review caching and
tailed logs, with model/embedding boundaries stubbed. Regression coverage includes both edges of
non-transitive close-pair chains and identical-run resume. No paid model calls were made; these
checks do not establish Sonnet's semantic accuracy.
