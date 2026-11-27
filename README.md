# agentic-hivemind

Hypothesis decomposition and feature deduplication for the Agentic Hivemind paper.
The current merge protocol follows the manager's [GitHub main branch](https://github.com/someshsingh22/agentic-hivemind/blob/main/ssh/dedup.py),
with source-column matching added as explicitly requested.

## Current protocol

- Sonnet decomposes each hypothesis into scalar per-row feature extractors and its frozen regression test.
- Features have `name`, `dtype`, `source_columns`, `definition`, `kind` and `spec`.
- Embed `name: definition` for **all features**, code and prompt alike, as on main.
- Merge with one complete-linkage cosine-distance clustering pass. Features from the same hypothesis,
  with different dtypes, or with different original source-column sets cannot merge. Source order is ignored.
- There is no Sonnet equivalence judge, approval matrix, alignment category or human-resolution step.
  Code features can merge with code or prompt features when the gates and distance allow it, as on main.
- Keep each hypothesis slot, direction and frozen formula separately; map feature references to canonical names.
- Keep prompt extraction specs unchanged. Code recipes use main's word-based canonical reference renaming.
  This is not a Python parser and retains main's limitations around names appearing in methods or literals.
- Retain structural/source validation and collision-safe canonical names. The custom dependency parser and
  bare-reference rejection have been removed.

This protocol measures embedding-based feature similarity, not verified identity of full measurement
functions. Matching sources does not make every close pair interchangeable. Earlier Sonnet-reviewed
results used a different protocol and must not be presented as the same evaluator.

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

The three trials share the same decomposition and one all-feature embedding batch per invocation.
Cutoffs remain 0.20, 0.25 and 0.30. The current output directory remains `runs/calibration_v7/cmv/`:

- `threshold_0.20/`, `threshold_0.25/`, `threshold_0.30/`: trial plans, effective configs and distance exports.
- `tuning_summary.json`: main-style hypothesis/raw/unique/prompt-feature counts.
- Earlier judged results under `automated/` remain untouched and are not consumed.

Without `--tune`, decomposition and the configured single cutoff write `plan.json` at the root, as on main.
The explicit `ssh test` command reads that plan and an externally populated feature CSV. Tuning does not
select or install a statistical-test plan automatically.

`feature_pairs.csv` covers all code and prompt pairs with original specs, distance, eligibility and final
merge state. `calibration_sample.csv` is a deterministic seed-42 audit sample without manual-label fields.
`threshold_sensitivity.csv` reports only total/eligible/ineligible/candidate pair counts, not accuracy.
These diagnostics do not change the clustering decisions.

## Local verification

```bash
uv run python -m unittest discover -s tests -v
```

Deterministic checks cover all-feature merging, source/dtype/owner gates, complete-linkage boundaries,
canonical code/formula references, file-exists reuse, interrupted-run resume and distance exports.
A Node 24+ check executes the submit extension. A throwaway comparison exercised the fetched main
clustering implementation and confirmed matching memberships when sources match. The actual nohup/tail
launcher was exercised with simulated model/embedding responses; no paid calls were made locally.
