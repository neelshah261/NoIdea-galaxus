# contracts/experiment.md: one EXP row in results/metrics.json

Owned by design; read-only for the lanes. `results/metrics.json` is written **only** by the model lane's eval script
(`python -m galaxus.model.eval`, through `galaxus.model.results.append_row`), never by hand (AGENTS.md product
rule 1). Every number shown to judges is one of these rows. Ids `EXP-####` are never reused or renumbered.

The fields below are the ones the eval script writes today (EXP-0001, EXP-0002).

## Identity and reproducibility

| Field | Meaning |
| --- | --- |
| `id` | `EXP-####` |
| `created` | ISO time with offset |
| `predictor` | the name registered in `galaxus.model.eval.PREDICTORS` |
| `commit` | the full git sha the row was produced from |
| `dirty` | `true` if the working tree had uncommitted changes; a judge-facing row must be `false` |
| `seed` | the seed (the holdout split itself is fixed: seed 0, challenge_description.md "Scoring locally") |
| `model`, `model_revision`, `docker_digest` | which weights and image produced it, or `null` for no model |
| `config` | the method's settings (backend, image size, context on/off, prompt hash, tokens, retries, workers, ...) |
| `note` | one line of context, for example why a figure is not a product figure |

## Scores (official holdout, through metric.py)

| Field | Meaning |
| --- | --- |
| `score` | mean macro F1 over the six dimensions: the leaderboard number |
| `macro_f1` | macro F1 per dimension |
| `overall_weighted_f1`, `weighted_f1` | weighted F1, overall and per dimension (the hosts look at it too) |
| `holdout` | `images`, `products`, `training_images` and the `sha256` of the holdout ids; rows with a different hash are never compared |

## Speed and cost

| Field | Meaning |
| --- | --- |
| `images_per_second` | measured over the whole holdout run |
| `usd_per_hour`, `cost_per_100k_usd` | `null` unless a price with a source is given; never estimated in the row |
| `hardware` | `system`, `machine`, `python`, `gpu`, `peak_gpu_memory_gb` |

Nothing in a row may contain competition data, labels, image ids, keys or the passphrase.
