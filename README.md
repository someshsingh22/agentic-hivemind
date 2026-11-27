# agentic-hivemind

Hypothesis decomposition and feature deduplication for the Agentic Hivemind paper.
The merge protocol retains complete-linkage clustering from the manager's
[GitHub main branch](https://github.com/someshsingh22/agentic-hivemind/blob/main/ssh/dedup.py),
but deliberately diverges through source-column matching and prompt-only semantic deduplication.

## Current protocol

- Sonnet decomposes each hypothesis into scalar per-row feature extractors and its frozen regression test.
- Features have `name`, `dtype`, `source_columns`, `definition`, `kind` and `spec`.
- Embed `name: definition` for **prompt features only**, in their original order.
- Merge prompts with one complete-linkage cosine-distance clustering pass. Features from the same hypothesis,
  with different dtypes, or with different original source-column sets cannot merge. Source order is ignored.
- Code features remain unique, even for identical recipes and at cosine cutoff 2; they are neither embedded
  nor included in cosine-pair exports. Code-only banks require no embedding request.
- There is no Sonnet equivalence judge, approval matrix, alignment category or human-resolution step.
- Keep each hypothesis slot, direction and frozen formula separately; map feature references to canonical names.
- Keep prompt extraction specs unchanged. Code recipes use main's word-based canonical reference renaming.
  This is not a Python parser and retains main's limitations around names appearing in methods or literals.
- Retain structural/source validation and collision-safe canonical names. The custom dependency parser and
  bare-reference rejection have been removed.

This protocol measures embedding-based prompt-feature similarity, not verified identity of full measurement
functions. Matching sources does not make every close prompt pair interchangeable. Excluding code prevents
semantic clustering from conflating distinct code measurements; it is not a semantic-accuracy fix for prompt
merges. Earlier all-feature and Sonnet-reviewed results used different protocols and must not be presented
as the same evaluator. Historical artifacts are unchanged; existing files do not acquire the new semantics.

## Extraction prompts and dataset context

The accepted decomposition prompt remains in `ssh/prompts.yaml`, using an ExperiGen-style YAML layout.
It directly requests the needed row-level semantic quantity, without mandatory comment-label vectors,
label classes or dependency fields. Code is used for deterministic measurements; explicit thresholds
may retain a base measurement and add a derived feature, as in manager main.

The latest prompt refinement adds three hypothetical boundary examples only: exactly one versus multiple
comments, any versus all normative arguments, and quotation versus quotation-and-refutation. They illustrate
how to preserve names, definitions and extraction rules; they do not add features, merge gates or model calls.
This is prompt tuning, not evidence that embedding distances now separate equivalent measurements.

`envs/cmv.yaml` provides the richer CMV description adapted from the ExperiGen balanced-2k configuration:
text scope, attribution, within-comment conjunctions, comments versus commenters, unavailable measurements,
numeric-field limitations and outcome-leakage precautions. These extraction instructions are unchanged
by the merge/cache restoration. The statistical evaluator, models and threshold values are unchanged too.

Leakage guards and rubric consistency remain instructions to Sonnet, not programmatic guarantees.
The raw CSV remains available to the decomposition agent; the target is not technically withheld.
The submit tool validates structural contracts before saving results. Annotation and SSH testing remain separate.

## Caching like main

If `decomposed/<hypothesis_id>.json` exists, reuse it; otherwise generate it. Cached features still undergo
structural/source validation. There is no dataset/prompt/model fingerprint gate or new input manifest.
Changing input values, hypotheses, prompts or models does not automatically invalidate existing files.
Use a new output directory explicitly when a fresh decomposition is intended.

Existing completed v7 per-hypothesis files can therefore be reused for this merge-only comparison.
Prior `decomposition_inputs.json` files are ignored, not removed. Existing artifacts are not relabeled.

## Run

Set `ANTHROPIC_FOUNDRY_RESOURCE=mdsr-resource-dev-resource` and `ANTHROPIC_FOUNDRY_API_KEY` in `.env`.
Use Node 24 LTS with `@earendil-works/pi-coding-agent` installed. Never commit credentials.

After any currently running worker has finished:

```bash
git pull --ff-only origin namit-rows
bash run_calibration.sh --tune
```

The launcher uses `/mnt/localssd/namit/experigen-evals/.venv`, starts a nohup worker and follows its log.
It runs decomposition and distance-based deduplication only. No pair-review Sonnet calls, row annotation,
Qwen calls or SSH tests run. Set `VENV` to another existing environment if needed.

The three trials share the same decomposition and one prompt-only embedding batch per invocation.
Cutoffs remain 0.20, 0.25 and 0.30. The current output directory remains `runs/calibration_v7/cmv/`:

- `threshold_0.20/`, `threshold_0.25/`, `threshold_0.30/`: trial plans, effective configs and distance exports.
- `tuning_summary.json`: main-style hypothesis/raw/unique/prompt-feature counts.
- Earlier judged results under `automated/` remain untouched and are not consumed.

Without `--tune`, decomposition and the configured single cutoff write `plan.json` at the root, as on main.
The explicit `ssh test` command reads that plan and an externally populated feature CSV. Tuning does not
select or install a statistical-test plan automatically.

`feature_pairs.csv` covers every unordered prompt pair with original specs, distance and final merge
state. No code pair distances or separate eligibility fields are exported.
`calibration_sample.csv` is a deterministic seed-42 sample of pairs passing the source/dtype/hypothesis
checks. `threshold_sensitivity.csv` reports total pair and candidate counts, not accuracy.
These diagnostics do not change clustering decisions. Source/dtype/hypothesis checks remain inline;
there is no separate eligibility state or exported eligibility column. Existing artifacts are unchanged.

### Compare short and full feature descriptions

This separate experiment uses the saved feature bank without running decomposition:

```bash
git pull --ff-only origin namit-rows
bash run_calibration.sh --compare-descriptions runs/calibration_v8_examples/cmv/decomposed.json
```

The launcher still uses nohup and follows the log. It makes one embedding request containing
both representations, in the same prompt-only feature order:

- `short_definition`: `name: definition`, matching the current default.
- `full_description`: `name: spec (dtype: dtype)`, using the existing complete prompt extraction
  instructions. No feature definitions or prompts are regenerated.

Both views embed only saved prompt features. All saved code features remain unique in both plans.
They use the same embedding configuration, complete-linkage merging and source/dtype/hypothesis gates,
at distances 0.20, 0.25 and 0.30.
This is a representation comparison, not a new equivalence judge or an automatic threshold choice.
Longer descriptions expose measurement rules but can also increase shared boilerplate; inspect
actual merged and missed pairs before selecting a setting. ExperiGen prod's description-based
search is candidate retrieval, not evidence that cosine similarity establishes equivalence.

Results are written beside the input file under `semantic_description_comparison/`:

- `inputs.json`: source path, embedding configuration, ordered prompt-feature IDs and exact embedding texts.
- `short_definition/threshold_*/` and `full_description/threshold_*/`: plans, configs and distance exports.
- `comparison_summary.json`: counts for both representations at each threshold.

Existing decomposition, baseline plans, historical `description_comparison/` artifacts and the default pipeline
are not overwritten. Each comparison invocation re-embeds both views and replaces only its own semantic
comparison outputs. No Sonnet generation,
pair reviews, row annotation or SSH tests run. Pass a different config as the final launcher argument
if needed; its embedding settings are used, while its output root and hypothesis sampling are ignored
because this command operates on the explicitly supplied saved bank.

## Local verification

```bash
uv run python -m unittest discover -s tests -v
```

Deterministic tests cover prompt-only merging, code uniqueness without embeddings, source/dtype/owner gates,
complete-linkage boundaries, canonical code/formula references, file-exists reuse, interrupted-run resume
and prompt-pair distance exports. Code indicators for exactly one versus at least two comments retain
distinct frozen terms and compute outputs. A Node 24+ check executes the submit extension.
Historical comparisons against the fetched main implementation predate prompt-only deduplication and
do not establish current all-feature parity.
After the bookkeeping cleanup, a CLI replay using saved embedding geometry reproduced all six semantic
partitions and candidate counts, with all 16 code features separate. No paid model calls were used.
