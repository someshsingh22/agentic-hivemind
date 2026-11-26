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
   Each feature has `name, dtype, source_columns, definition, kind=code|prompt, spec, class,
   annotation_unit, depends_on`, plus a *relationship*: H0/H1, an OLS formula, tested `term`, claimed `sign`.
   `source_columns` lists original predictors, including all jointly read columns; it excludes the outcome and
   identifiers. The submission schema restricts these names to the actual CSV header.
   Primary features measure original inputs (`depends_on: []`). Semantic primary features have a
   self-contained annotation rubric. `annotation_unit: row` returns a scalar; `comment` returns a list
   aligned to this row's original `OC_comments` slots, optionally reading `title`/`argument` as context.
   For comment outputs, `dtype` describes each list element. Missing/unassessable items are null.
   Secondary features are row-level code over named dependencies: any/count/proportion/mean/threshold.
   Their original source columns must cover their dependencies' sources; cyclic/unknown dependencies fail.
   Thus any/example-count/example-share can share one semantic comment predicate while remaining distinct
   final variables. This preserves the 0-10 intensity versus density distinction, not a generic emotion proxy.
   Editable prompts live in `ssh/prompts.yaml` under `agents.<name>.prompt_config.system_message` and
   `user_template`, following ExperiGen's YAML layout. Both use a shared `measurement_policy` for
   missingness, empty inputs, attribution and output alignment; user templates use `$variable` substitution.
   Definitions must preserve named comparison groups, frequency, distinctness and temporal attribution;
   an unfaithful proxy or a contrast not representable by one coefficient gets `term: null`.
   Sessions and the dataset-specific submission schema are saved in `<out>/<env>/pi/`.
2. **dedup**: embeds `name: definition` for **semantic features only** and runs complete-linkage candidate
   clustering at cosine distance `0.25`. Different source-column sets/dtypes/annotation units and
   same-hypothesis features cannot merge. Source order is ignored; `argument` and `title + argument`
   are different source sets. Only semantic primary measurements are deduplicated.
   Sonnet then compares **every pair of full rubrics within each multi-feature candidate cluster** using
   the YAML `feature_equivalence` prompt. Review sessions have only `submit`, no data/code tools.
   Only explicit `equivalent` decisions permit merging. `distinct`, `uncertain`, and `needs_alignment`
   stay separate. `needs_alignment` identifies a shared construct whose operational boundary must be
   resolved; it does not rewrite a rubric or approve a merge. Paraphrased explanations are not differences
   by themselves; real scope, quantifier, scale, attribution or temporal differences still block merging.
   Every pair in a final cluster must be approved, so A=B and B=C cannot override a rejected A=C.
   A malformed or incomplete review blocks plan creation; it never silently approves or guesses a merge.
   Every code feature stays separate and is never sent for embedding or equivalence review.
   `plan.json` preserves canonical features, original members, frozen tests and `dedup_reviews` with full
   pair decisions/reasons and distances. Code references and `depends_on` are rewritten to canonical names.
   `feature_pairs` includes every semantic pair, even ineligible or outside the candidate clusters, so
   missed reuse can be inspected. Distances are exported without additional embedding requests.
   Reviews are cached under `<out>/<env>/dedup_reviews/` by full feature rubrics, resolved prompts, schema
   and model configuration; sessions live in `pi_review/`. Rubric/prompt changes cannot reuse an approval.
   This adds Sonnet calls per multi-feature candidate cluster; it is not dataset-row annotation.
   LLM review is still fallible: inspect merges before annotation. Code computation, row annotation and
   SSH testing remain outside this command.
3. **test**: fits OLS with cluster-robust SEs by the env `group` (HC1 if there is none). It applies Holm correction
   across testable hypotheses. `ssh` means `p_holm <= alpha` and the coefficient sign matches the claim.

All parameters are in `config.yaml`: models, thinking effort, embedding model, threshold, alpha, seed, and sample size.
Datasets are in `envs/<env>.yaml`: target, group, CSV, hypotheses JSON, and a column description.
Secrets are read from env vars (see `.env.example`). Set `ANTHROPIC_FOUNDRY_RESOURCE` to
`mdsr-resource-dev-resource` and `ANTHROPIC_FOUNDRY_API_KEY` to the actual resource key.
Both Sonnet and the embedding endpoint use these names; the old `AZURE_*` names are not used.

```bash
npm i -g @earendil-works/pi-coding-agent     # use Node 24 LTS, not Node 18
cp .env.example .env                        # fill ANTHROPIC_FOUNDRY_API_KEY; never commit .env
bash run_calibration.sh
```

## First calibration: 100 CMV hypotheses

On the experiment machine, place the supplied inputs at the paths already configured in `envs/cmv.yaml`:

| Supplied file | Repository-relative destination |
|---|---|
| `cmv_random_balanced_2000.csv` | `data/cmv.csv` |
| `ssh_100.json` | `data/hypotheses_ssh_eval/cmv.json` |

Alternatively, edit the two input paths in `envs/cmv.yaml`. Run commands from the repository root.
The JSON's `hypotheses` list already contains 100 entries; leave `n_hypotheses: null` to keep all of them.
Decomposition and full-rubric equivalence review use Sonnet 5.5; candidate embeddings use `text-embedding-3-large`.
The default is one concurrent Sonnet session (`agent.workers: 1`) to reduce bursts against the shared
output-token quota. Sequential sessions can still be rate-limited; this is not a tokens-per-minute limiter.
Changing the setting does not affect an already running worker. After a failed run has fully exited, rerun
the launcher to reuse completed per-hypothesis JSON files rather than deleting the run directory.
Use the detached launcher for the calibration. It uses the existing GPU-machine environment at
`/mnt/localssd/namit/experigen-evals/.venv`, not a new repo-local venv:

```bash
bash run_calibration.sh
```

The script runs decomposition followed by deduplication under `nohup` and automatically follows its timestamped
log in `runs/launchers/` with `tail`. It prints the worker PID, log path, and command to follow the log again.
Ctrl-C stops only the log follower; the worker continues after disconnect. Do not launch another copy while it
is running. If decomposition fails, deduplication does not start. While attached, the launcher returns the
worker's exit status when the run ends.

Set `VENV` to use another existing environment, or pass environment/config paths as positional arguments:

```bash
VENV=/mnt/localssd/namit/experigen-evals/.venv bash run_calibration.sh envs/cmv.yaml config.yaml
```

The launcher calls `python -u -m ssh` directly; it does not use `uv run`, install packages, or change the model
server. Ensure the shared venv has the dependencies in `pyproject.toml`, and `pi` is on PATH.
These two stages do not call Qwen, annotate the 2,000 dataset rows, or run SSH tests.

The default `out: runs/calibration_v3` writes artifacts to `runs/calibration_v3/cmv/`. Both prior
calibrations (`runs/cmv/`, `runs/calibration_v2/cmv/`) remain untouched; v3 requires new decomposition because
its feature contract changed. Return `decomposed.json`, `plan.json`, `decomposition_inputs.json`,
`dedup_reviews/` and the generated calibration CSVs for local inspection.
The plan's `members` retain original definitions; `dedup_reviews` retain decisions and reasons.
Review session logs before publishing them; never include `.env` or other credentials.
Do not merge hypotheses or discard repeated hypothesis slots.

Use a fresh `out` directory when changing the decomposition prompt/schema, model configuration, dataset or
hypothesis bank. `decomposition_inputs.json` fingerprints the actual CSV, exact hypotheses, environment,
agent settings, schema and resolved prompts; changed inputs or unversioned legacy caches are rejected before
agents start. Identical interrupted runs reuse completed results. Old calibrations are not silently migrated.
The pi launcher resolves schema/output/session paths before switching to its session directory; relative
`out` paths work for both decomposition and review. A launch-path fix alone does not invalidate decompositions.
Run only decomposition and reviewed deduplication for now; annotation and SSH testing come later.

## Human calibration of cosine distance

Deduplication also writes:

| File | Contents |
|---|---|
| `feature_pairs.csv` | All semantic pairs with original full rubrics, sources/units, distance, eligibility, candidate membership, model decision and final merge state. |
| `calibration_sample.csv` | At most 10 eligible pairs per distance bin, sampled with seed 42, including distances above 0.25. |
| `threshold_sensitivity.csv` | Candidate counts for cutoffs 0.10–0.40; quality metrics blank until explicit human labels exist. |

These are generated outputs; copy the sample before labeling it, so rerunning deduplication cannot overwrite
your review. Fill `review_label` with `equivalent`, `distinct` or `uncertain`, and `review_reason_human` with
the reason. Leave unreviewed pairs blank. Model verdicts are never used as human labels. The annotation-sharing
question is whether the SAME base predicate can be reused; do not merge different predicates merely because
their outputs happen to correlate.

```bash
cp runs/calibration_v3/cmv/calibration_sample.csv runs/calibration_v3/cmv/reviewed_sample.csv
# Review/edit reviewed_sample.csv, then use the activated evaluation venv:
nohup python -u -m ssh calibrate \
  runs/calibration_v3/cmv/reviewed_sample.csv \
  runs/calibration_v3/cmv/human_calibration \
  > runs/calibration_v3/cmv/human_calibration.log 2>&1 < /dev/null &
PID=$!
tail --pid="$PID" -n +1 -F runs/calibration_v3/cmv/human_calibration.log
```

Threshold sensitivity describes only the human-reviewed eligible subset, not population precision/recall,
and does not automatically change `0.25`. It measures pairwise candidate retrieval, not final clustering
accuracy. Review the full eligible distribution and missed equivalents before selecting a new cutoff.
Compare unique semantic annotation tasks separately from total final variables; secondary code features
can grow without adding model annotation tasks. No savings or annotation accuracy is established until reviewed.

## Local verification

```bash
uv run python -m unittest discover -s tests -v
```

The suite uses fixed/seeded embeddings and simulated agent submissions, with no model or embedding API calls.
It checks source/unit matching, semantic-only merges, separate code aggregations and canonical dependencies,
complete-linkage boundaries, rejected/uncertain/alignment-needed pairs, complete review coverage, cache identity,
and deterministic distance sampling with explicitly human-labeled threshold scoring.
A local CLI smoke also exercised decomposition/resume, deduplication, pair exports and label scoring with
external model/embedding boundaries stubbed. Neither check establishes Sonnet's semantic accuracy.
