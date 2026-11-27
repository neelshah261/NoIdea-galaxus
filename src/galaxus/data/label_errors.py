"""Suspected label errors in ``train.csv``, for the bonus.

[challenge_description.md, "How we assess"](../../../challenge_description.md#how-we-assess) and [FAQ.md,
"Hand-in and assessment"](../../../FAQ.md#hand-in-and-assessment) ask for a csv with exactly the columns
``image_id, dimension, label, suggested_label``, one row per suspected error, sent by direct message to the Galaxus
hosts on Discord by 14:30 on 28 November.

It holds labels, so it never enters git: it is written to ``$GALAXUS_DATA/out/`` and a human sends it. Agents do not
message anyone on Discord (AGENTS.md "Galaxus product rules" 7).

Only disagreements with a **human** label are reported. A ``model`` prelabel is not a label error: about a fifth of
the train cells are unchecked prelabels (FAQ.md, "Why do some train labels have the source ``model``?").

The out-of-fold format is design's, not this lane's: PLAN.md "Lanes and contracts" (branch ``design/3-plan``) and
``contracts/prediction.md`` once B0 (#11) lands define ``Prediction{image_id, labels: {dimension: class},
confidence: {dimension: float} | None}``. The model lane writes exactly those records, one JSON object per line, to
``$GALAXUS_DATA/out/predictions/<EXP id>/holdout.jsonl`` (``galaxus.model.eval.write_predictions``).

:func:`out_of_fold_frame` therefore reads, in this order of preference:

1. a path to that jsonl file, or to the directory holding it;
2. ``Prediction`` records, as mappings or as objects with ``image_id`` / ``labels`` / ``confidence``;
3. a long table with the columns ``image_id, dimension, prediction, confidence``;
4. ``{image_id: {dimension: (class, confidence)}}``.

Forms 3 and 4 predate the contract and are kept because they are the convenient ones to write by hand in a test.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

from galaxus.data.classes import ID_COLUMN, dimensions
from galaxus.data.paths import out_dir

#: The columns the hosts expect, in this order. Do not add any.
REPORT_COLUMNS = (ID_COLUMN, "dimension", "label", "suggested_label")
DEFAULT_NAME = "label_errors.csv"
DEFAULT_THRESHOLD = 0.9

PREDICTION_COLUMNS = (ID_COLUMN, "dimension", "prediction", "confidence")

#: The file the model lane's eval writes its holdout answers to, one ``Prediction`` record per line.
PREDICTIONS_FILE = "holdout.jsonl"

OutOfFold = pd.DataFrame | Mapping[str, Mapping[str, object]] | Iterable[object] | str | os.PathLike[str]


def _records_from_jsonl(path: str | os.PathLike[str]) -> list[object]:
    """The model lane's ``holdout.jsonl``, or the directory holding it. It lives outside the repository."""
    target = Path(path)
    if target.is_dir():
        target = target / PREDICTIONS_FILE
    if not target.is_file():
        raise FileNotFoundError(f"no out-of-fold predictions at {target}: the model lane writes {PREDICTIONS_FILE}")
    with target.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rows_from_records(records: Iterable[object]) -> list[tuple[str, str, str, object]]:
    """Rows from ``Prediction`` records, mappings or objects alike."""
    rows: list[tuple[str, str, str, object]] = []
    for record in records:
        if isinstance(record, Mapping):
            image_id, labels, confidence = record["image_id"], record["labels"], record.get("confidence")
        else:
            image_id, labels = record.image_id, record.labels
            confidence = getattr(record, "confidence", None)
        if confidence is None:
            raise ValueError(
                f"the prediction for {image_id!r} has no confidence, so it cannot be ranked: the finder reports "
                "high-confidence disagreements only (issue #7). Fill Prediction.confidence in the eval run."
            )
        for dimension, prediction in (labels or {}).items():
            if dimension not in confidence:
                raise KeyError(f"{image_id!r} predicts {dimension!r} but carries no confidence for that dimension")
            rows.append((str(image_id), str(dimension), str(prediction), confidence[dimension]))
    return rows


def out_of_fold_frame(predictions: OutOfFold) -> pd.DataFrame:
    """Out-of-fold predictions as a long table with the columns in :data:`PREDICTION_COLUMNS`.

    Accepts any of the four forms listed in the module docstring.
    """
    if isinstance(predictions, (str, os.PathLike)):
        predictions = _records_from_jsonl(predictions)
    if isinstance(predictions, pd.DataFrame):
        missing = [column for column in PREDICTION_COLUMNS if column not in predictions.columns]
        if missing:
            raise KeyError(f"the out-of-fold table is missing {missing}; expected columns {list(PREDICTION_COLUMNS)}")
        frame = predictions[list(PREDICTION_COLUMNS)].copy()
    elif isinstance(predictions, Mapping):
        rows = []
        for image_id, per_dimension in predictions.items():
            for dimension, value in (per_dimension or {}).items():
                if isinstance(value, Mapping):
                    prediction, confidence = value["prediction"], value["confidence"]
                else:
                    prediction, confidence = value
                rows.append((str(image_id), str(dimension), str(prediction), confidence))
        frame = pd.DataFrame(rows, columns=list(PREDICTION_COLUMNS))
    else:
        frame = pd.DataFrame(_rows_from_records(predictions), columns=list(PREDICTION_COLUMNS))
    frame[ID_COLUMN] = frame[ID_COLUMN].astype(str)
    frame["dimension"] = frame["dimension"].astype(str)
    frame["prediction"] = frame["prediction"].astype(str)
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
    unusable = frame["confidence"].isna()
    if unusable.any():
        raise ValueError(f"{int(unusable.sum())} out-of-fold row(s) have a confidence that is not a number")
    return frame


def find_label_errors(
    predictions: OutOfFold,
    train: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    dimension_names: tuple[str, ...] | None = None,
    scored_only: bool = False,
) -> pd.DataFrame:
    """High-confidence disagreements with a human train label, most confident first.

    Args:
        predictions: out-of-fold predictions with confidences, see :func:`out_of_fold_frame`.
        train: ``train.csv`` as read by :func:`galaxus.data.load.read_train`.
        threshold: report a disagreement only when the prediction's confidence is at least this.
        dimension_names: which dimensions to look at. Defaults to all six.
        scored_only: also require the cell to be relevant, i.e. restrict to the cells the metric scores. Off by
            default, because a human label in an unscored cell can still be wrong.

    Returns:
        A table with the columns in :data:`REPORT_COLUMNS`, ranked by confidence. One row per suspected error.
    """
    names = tuple(dimension_names) if dimension_names is not None else dimensions()
    frame = out_of_fold_frame(predictions)
    frame = frame[frame["dimension"].isin(names) & (frame["confidence"] >= threshold)]

    labels = []
    for dimension in names:
        source = f"{dimension}_source"
        if dimension not in train.columns or source not in train.columns:
            raise KeyError(f"train.csv has no {dimension}/{source} column: human label errors cannot be found")
        human = train[train[source] == "human"]
        if scored_only:
            from galaxus.data.split import scored_mask

            human = train[scored_mask(train, dimension)]
        labels.append(
            pd.DataFrame(
                {
                    ID_COLUMN: human[ID_COLUMN].astype(str),
                    "dimension": dimension,
                    "label": human[dimension].astype(str),
                }
            )
        )
    human_labels = (
        pd.concat(labels, ignore_index=True) if labels else pd.DataFrame(columns=[ID_COLUMN, "dimension", "label"])
    )

    merged = frame.merge(human_labels, on=[ID_COLUMN, "dimension"], how="inner")
    disagreements = merged[merged["label"] != merged["prediction"]].copy()
    disagreements = disagreements.rename(columns={"prediction": "suggested_label"})
    disagreements = disagreements.sort_values(
        ["confidence", ID_COLUMN, "dimension"], ascending=[False, True, True], kind="stable"
    )
    return disagreements[list(REPORT_COLUMNS)].reset_index(drop=True)


def write_label_errors(
    report: pd.DataFrame,
    path: str | os.PathLike[str] | None = None,
    name: str = DEFAULT_NAME,
) -> Path:
    """Write the report to ``$GALAXUS_DATA/out/`` and return the path. This file never enters git."""
    columns = list(REPORT_COLUMNS)
    if list(report.columns) != columns:
        raise ValueError(f"the report must have exactly the columns {columns}, got {list(report.columns)}")
    target = Path(path) if path is not None else out_dir() / name
    target.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(target, index=False)
    return target
