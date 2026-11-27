# Grouped Sonnet-only deduplication

Branch: `namit-sonnet-only`. This separate workflow reads the full frozen saved
feature bank; it does not change the existing `ssh` embedding pipeline. It runs
sequential grouped Sonnet reviews only: no decomposition, embeddings, row
annotation, alignment/human-resolution stage, feature rewriting or SSH statistics.
Run paid reviews on the remote machine only.

## Hybrid pair-review experiment

The isolated hybrid command reuses the saved paper-definition bank and pair distances.
It does not replace the embedding-only or grouped Sonnet-only workflows.

```bash
git pull --ff-only origin namit-sonnet-only
bash sonnet_dedup/run_hybrid.sh
```

The foreground runner activates the evaluation venv, then Node 24. Keep the terminal open.
It reviews 247 compatible pairs at retrieval distance <= 0.30 in 25 sequential Sonnet
sessions of up to 10 pairs. This is a candidate cutoff, not a merge threshold. Source-set,
dtype and different-hypothesis checks remain. Code features stay separate.

Sonnet receives both complete feature definitions/specs, without cosine scores, and must
try to find an input where their outputs differ. Each pair gets a boolean equivalence
decision and explanation. Missing candidate pairs count as unapproved. A deterministic
greedy partition in original feature order joins a group only if every member pair has
explicit approval; it does not close transitive chains. The partition need not maximize
merges, and independently approved pairs may remain separate due to another group member.

Inputs: `runs/calibration_paper_definitions_100/cmv/decomposed.json` and `plan.json`.
Outputs: `runs/hybrid_paper_definitions/cmv/inputs.json`, `batch_*.response.json`,
`decisions.json`, `summary.json`, and `plan.json`. The plan retains all saved pair distances
and new group memberships; `within_threshold` refers to the retrieval cutoff. Original
inputs/results are untouched. Completed batches are reused; changed inputs, model or prompt
require a new output directory. Never commit the generated `pi/` credentials/session directory.

No decomposition, new embeddings, row annotation or statistical tests run. This experiment
makes paid Sonnet calls on the GPU machine; decision accuracy remains unverified until reviewed.
An offline smoke exercised the actual CLI on all 247 candidates with a simulated agent,
verified the only simulated-approved pair merged, preserved all 128 feature contracts and
100 claims, kept 21 code features separate, and resumed with zero additional agent calls.
Tests cover nontransitive grouping and malformed/incomplete decisions.


## Remote launch

After any existing worker has finished, from the repository root on the remote:

```bash
git fetch origin namit-sonnet-only
git switch namit-sonnet-only
git pull --ff-only origin namit-sonnet-only
bash sonnet_dedup/run.sh
```

The defaults are equivalent to:

```bash
bash sonnet_dedup/run.sh \
  runs/calibration_v8_examples/cmv/decomposed.json \
  config.yaml \
  runs/sonnet_only/cmv
```

The launcher uses the existing `/mnt/localssd/namit/experigen-evals/.venv`;
set `VENV` to another existing environment directory if needed. It starts a
`nohup` worker and follows its log with `tail --pid`. Ctrl-C stops following,
not the worker. The printed log path can be followed again; do not start a
second worker against the same output directory while one is running.
Launcher logs live in `runs/launchers` (override with `LOG_DIR`).

The CLI loads the repository-root `.env`. Configure the same Foundry Sonnet
credentials used by the existing pipeline: `ANTHROPIC_FOUNDRY_RESOURCE` and
`ANTHROPIC_FOUNDRY_API_KEY`. Node 24 LTS and `@earendil-works/pi-coding-agent`
are required as in the existing workflow. Never commit credentials.

## Review protocol

For the saved bank of 117 prompt features, there are ten source-column/dtype
batches, of which eight require Sonnet sessions because they have cross-owner
pairs. This is approximately eight grouped sessions, not a separate judgment
for every feature pair. Each session receives full feature specifications,
including names, definitions and extraction instructions. All 16 code features
remain unique and are not submitted for semantic merging.

Sonnet groups prompt features by equivalence of the full measurement. Ordinary
paraphrases may merge; materially different operations must not. Source-column
sets and dtypes must match, and a merged group cannot contain two features
from the same hypothesis. These are the existing source/type/different-owner
rules, not an additional eligibility classification.

Responses contain `groups`, each with `members` (opaque keys such as
`h00000:feature_name`) and a `reason`. Every prompt feature in a batch must
appear exactly once in a nonempty group. The submit-only session tool validates
responses before saving, and cached responses are validated again on load.
Groups and their reasons are auditable model decisions, not gold-truth labels
or evidence of semantic accuracy. Canonical naming and code-reference remapping
follow the existing deduplication conventions; prompt specs are not rewritten.

## Outputs and resume

The output directory (default `runs/sonnet_only/cmv`) contains:

- `plan.json`: canonical features and remapped hypothesis references.
- `summary.json`: deduplication summary.
- `decomposed.json`: the frozen input-bank snapshot.
- `prompts.yaml`: a copy of the review prompts.
- `config.json`: effective configuration.
- `groups/000.json`, etc.: saved batch inputs.
- `groups/000.schema.json`, etc.: response schemas for those batches.
- `groups/000.response.json`, etc.: grouping responses and reasons.
- `pi/sessions/`: pi session records.

Existing response files are reused on resume, like the main workflow's
file-existence cache. Missing responses are generated sequentially. There is
no automatic bank/prompt/model fingerprint invalidation: use a fresh output
directory whenever the feature bank, prompts or model change. Do not modify
the original saved decomposition or reuse an old output directory for a
different experiment.

## Local verification

The full suite passed 41 tests, including the actual TypeScript submit extension.
A deterministic smoke exercised this launcher on the frozen 100-hypothesis bank
with simulated grouping responses: eight submit-only sessions, no embedding calls,
all 16 code features separate, full cache reuse, and one-session recovery after a
missing response. No paid calls were made. This verifies execution and preservation,
not Sonnet's semantic accuracy; inspect the real grouped output after the remote run.
