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
   The answer has reusable *features* (`name, dtype, source_columns, definition, kind=code|prompt, spec`)
   and a *relationship*: H0/H1, an OLS formula, the tested `term`, and the claimed `sign`.
   `source_columns` lists original predictors, including all jointly read columns; it excludes the outcome and
   identifiers. The submission schema restricts these names to the actual CSV header.
   For semantic (`prompt`) features, `spec` is a self-contained **per-row feature-extractor description**:
   source/speaker scope, inclusion/exclusion rules, scalar output and scale anchors, comment aggregation and
   denominator where applicable, and missing/unassessable handling. It measures a construct, not whether the
   hypothesis is true. Intensity qualifiers must survive decomposition; density is not intensity.
   Sessions and the dataset-specific submission schema are saved in `runs/<env>/pi/`.
2. **dedup**: embeds `name: definition` for **semantic features only** and runs complete-linkage clustering on
   cosine distance. Only features with the **same source-column set and dtype**, from different hypotheses,
   can merge. Source order is ignored, but `argument` and `title + argument` are different source sets.
   Every code feature stays separate and is never sent to the embedding service. Formulas and derived code
   references are rewritten to canonical names. Output: `plan.json`, containing the canonical feature registry,
   all original member definitions/specs/source columns, and every hypothesis's frozen test.
   The default `0.25` distance is a **provisional calibration threshold, not proof of semantic equivalence**.
   Same-source measurements can still differ in speaker, construct, intensity, scale or aggregation; inspect
   proposed merges before annotation. Code computation, row annotation and semantic-equivalence judging are
   not executed by this command.
3. **test**: fits OLS with cluster-robust SEs by the env `group` (HC1 if there is none). It applies Holm correction
   across testable hypotheses. `ssh` means `p_holm <= alpha` and the coefficient sign matches the claim.

All parameters are in `config.yaml`: models, thinking effort, embedding model, threshold, alpha, seed, and sample size.
Datasets are in `envs/<env>.yaml`: target, group, CSV, hypotheses JSON, and a column description.
Secrets are read from env vars (see `.env.example`). Set `ANTHROPIC_FOUNDRY_RESOURCE` to
`mdsr-resource-dev-resource` and `ANTHROPIC_FOUNDRY_API_KEY` to the actual resource key.
Both Sonnet and the embedding endpoint use these names; the old `AZURE_*` names are not used.

```bash
npm i -g @earendil-works/pi-coding-agent
cp .env.example .env                        # fill ANTHROPIC_FOUNDRY_API_KEY; never commit .env
uv run python -m ssh decompose envs/cmv.yaml
uv run python -m ssh dedup envs/cmv.yaml
uv run python -m ssh test envs/cmv.yaml annotated.csv   # annotated.csv has one column per plan feature
```

## First calibration: 100 CMV hypotheses

On the experiment machine, place the supplied inputs at the paths already configured in `envs/cmv.yaml`:

| Supplied file | Repository-relative destination |
|---|---|
| `cmv_random_balanced_2000.csv` | `data/cmv.csv` |
| `ssh_100.json` | `data/hypotheses_ssh_eval/cmv.json` |

Alternatively, edit the two input paths in `envs/cmv.yaml`. Run commands from the repository root.
The JSON's `hypotheses` list already contains 100 entries; leave `n_hypotheses: null` to keep all of them.
Decomposition uses the configured Sonnet 5.5 pi agent; deduplication uses `text-embedding-3-large`.
These two commands do not call Qwen or annotate the 2,000 dataset rows:

```bash
uv run python -m ssh decompose envs/cmv.yaml
uv run python -m ssh dedup envs/cmv.yaml
```

Review and return `runs/cmv/decomposed.json`, `runs/cmv/plan.json`, and the pi sessions under
`runs/cmv/pi/sessions/`. The plan's `members` preserve the raw measurements for reviewing each proposed merge.
Do not merge hypotheses or discard repeated hypothesis slots.

Use a fresh `out` directory in `config.yaml` when changing the decomposition prompt/schema or starting a new
calibration version; per-hypothesis results are cached. Old outputs without `source_columns` are rejected,
not guessed or silently migrated. If `out` is changed, artifacts live under `<out>/cmv/`.
Run only decomposition and deduplication for this first review; annotation and SSH testing come later.

## Local verification

```bash
uv run python -m unittest discover -s tests -v
```

The suite uses fixed/seeded embedding fixtures and cached decompositions, with no model or embedding API calls.
It checks source matching, semantic-only merges, preserved code features and references, complete-linkage
boundaries, and invalid source metadata. It does not validate Sonnet's semantic judgments.
