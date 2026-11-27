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

Classification follows the simple computation boundary in
[ExperiGen prod](https://github.com/someshsingh22/experigen/blob/256101d80d4b2dab0981aa896a5263663cd477d5/common_config.yaml#L79-L98):
code for counts, regex, arithmetic and fixed recipes; prompt for meaning, tone, intent and reasoning.
Both named comparison groups and the requested quantity must survive decomposition.

Our source-suitability policy is separate from that reference boundary. Raw-text code features
do not acquire mandatory semantic outcome filtering; they disclose contamination and timing risks
in `note`. Explicitly requested exclusions remain part of the measurement. Semantic specs still
exclude outcome reports as evidence of the construct. Raw word count is therefore not the same
measurement as the previous semantically filtered count; compare runs accordingly.

`envs/cmv.yaml` describes source scope, attribution, unavailable measurements and these source risks.
The feature schema, direct per-row extraction, statistical evaluator, models, merge rules and distance
thresholds are unchanged. These prompt instructions do not establish measured classification accuracy
or source suitability. The raw CSV remains accessible to the decomposition agent; the target is not
technically withheld. Structural submission validation does not establish scientific correctness.
A simulated-agent smoke exercised fresh prompt delivery, code-feature submission and cached restart;
all 41 existing tests passed. No paid model calls were used; model compliance remains unverified.

## Caching like main

If `decomposed/<hypothesis_id>.json` exists, reuse it; otherwise generate it. Cached features still undergo
structural/source validation. There is no dataset/prompt/model fingerprint gate or new input manifest.
Changing input values, hypotheses, prompts or models does not automatically invalidate existing files.
Use a new output directory explicitly when a fresh decomposition is intended.

Existing per-hypothesis files remain preserved. The full paper-style description run uses a fresh
output directory for all 100 hypotheses, separate from the earlier ten-hypothesis check.
Prior `decomposition_inputs.json` files are ignored, not removed.

## Run

Set `ANTHROPIC_FOUNDRY_RESOURCE=mdsr-resource-dev-resource` and `ANTHROPIC_FOUNDRY_API_KEY` in `.env`.
Use Node 24 LTS with `@earendil-works/pi-coding-agent` installed. Never commit credentials.

For all 100 hypotheses, after any running worker has finished:

```bash
git pull --ff-only origin namit-sonnet-only
bash run_decomposition.sh
```

The runner activates the evaluation venv before selecting Node 24, checks Node/pi availability,
then runs decomposition only. Override `VENV` or `NVM_DIR` if their locations differ.
It runs in the foreground; keep the terminal open. It does not launch a detached worker.
The configured output is `runs/calibration_paper_definitions_100/cmv/decomposed.json`.
It is decomposition plus classification, not relabeling of a frozen bank.
No embeddings, merging, annotation or statistical tests run.
Existing outputs are preserved. Completed per-hypothesis files are reused on restart.
The agent's stdout is suppressed; completed files appear in `decomposed/`.
A local smoke exercised the actual runner from another directory and the actual CLI's 100-hypothesis
selection/output, with Node/pi setup and the agent boundary simulated. No paid model calls were made.

The separate `bash run_calibration.sh` launcher runs decomposition AND deduplication, so do not
use it for classification-only review. It uses `/mnt/localssd/namit/experigen-evals/.venv`, starts
a nohup worker and follows its log. Set `VENV` to another existing environment if needed.
Its configured 0.20 cutoff remains an unvalidated working baseline; code features remain separate.

Optional `--tune` still compares 0.20, 0.25 and 0.30 using one shared decomposition and embedding batch.
It writes `threshold_*/` plans and `tuning_summary.json`; the normal single-cutoff run writes `plan.json`,
`feature_pairs.csv`, `calibration_sample.csv` and `threshold_sensitivity.csv` directly in the run directory.

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

### Original descriptions written during decomposition

The decomposition prompt requests `definition` as a short, plain-language question, following
[ExperiGen Figure 17 and §14.2.1](https://arxiv.org/pdf/2602.07983#page=50).
For example: "Does at least one comment validate the OP's dignity or effort before correcting
their reasoning in that same comment?" Numeric questions state the requested quantity, such as
"What proportion of words across the counterargument comments are content words?"
Keep output coding in `dtype`/`spec`, not "1 if ... else 0" in the description. Preserve decisive
conditions and quantity; full extraction rules remain in `spec`. This changes generated output,
not just the HTML display, without adding schema fields or a second description-generation call.
A simulated fresh/cached decomposition smoke and all 41 existing tests passed; real model
adherence to the new wording remains unverified. No paid calls were made.

The earlier description-writing revision did not change measurement rules. The current
classification revision additionally separates raw mechanical computation from source suitability.
The existing pipeline still embeds `name: definition` for prompt features only.
No second description, rewriting pass or additional Sonnet call is added.

Use the classification-only command above for the current review. File-exists caching is unchanged;
use a fresh output directory when changing prompts. Historical definition, faithfulness and
embedding-input results are preserved; the separate description-rewriting command is removed.

The previous experiment introduced incorrect merges even when its descriptions retained the
decisive differences. This prompt revision is not evidence that cosine matching is now correct.
Review the new definitions and actual false/missed merges before choosing a setting.

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
