# contracts/prediction.md: items in, predictions out

Owned by design; read-only for the lanes. The code is [`galaxus.types`](../src/galaxus/types.py). Dimension and
class names are never written here or in lane code: they come from [`classes.json`](classes.json) through
`galaxus.data.classes` (`dimensions()`, `allowed_classes()`).

## In: `ImageItem`

| Field | Type | Meaning |
| --- | --- | --- |
| `image_id` | `str` | the id from the competition table |
| `path` | `str` or `Path` | the image file (the data lane's prepared JPEG for model input) |
| `context` | `Mapping[str, str]` or `None` | product context from the catalog; `None` when every context cell is empty. A predictor must work without it ([challenge_description.md, "The task"](../challenge_description.md#the-task)) |

Built by `galaxus.serve.items.items_from_frame` (runner, submit, final run) and `galaxus.model.eval.items_from_rows`
(eval script).

## The method: `Predictor`

```python
def predict(self, batch: Sequence[ImageItem]) -> list[Prediction]: ...
```

One `Prediction` per item, for the same image ids. Any object with this method is a predictor (a runtime-checkable
protocol). The runner, the eval script, the benchmark and the Saturday script call nothing else, so a new method
registers itself and changes no other code.

## Out: `Prediction`

| Field | Type | Rule |
| --- | --- | --- |
| `image_id` | `str` | the item's id |
| `labels` | `Mapping[str, str]` | every dimension, one allowed class each: never empty, never unknown, `none` lowercase |
| `confidence` | `Mapping[str, float]` or `None` | probability of the chosen class per dimension, in [0, 1]; `None` when the method has none |

The data lane's validator enforces the `labels` rule on every file before it goes to Kaggle (AGENTS.md product
rule 2), following [challenge_description.md, "Submission file"](../challenge_description.md#submission-file).

## On disk: predictions JSONL

One JSON object per line, one line per image, written by the runner and `galaxus.model.eval.write_predictions` to
`$GALAXUS_DATA/out/predictions/<EXP id>/<part>.jsonl` (never into git):

```json
{"image_id": "<id>", "labels": {"<dimension>": "<class>", "...": "..."}, "confidence": {"<dimension>": 0.0} | null}
```

The runner appends and skips ids already written, so a run resumes after a crash without duplicates.
