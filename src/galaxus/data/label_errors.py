"""Suspected label errors in ``train.csv``, for the bonus.

[challenge_description.md, "How we assess"](../../../challenge_description.md#how-we-assess) and [FAQ.md,
"Hand-in and assessment"](../../../FAQ.md#hand-in-and-assessment) ask for a csv with exactly the columns
``image_id, dimension, label, suggested_label``, one row per suspected error, sent by direct message to the Galaxus
hosts on Discord by 14:30 on 28 November.

It holds labels, so it never enters git: it is written to ``$GALAXUS_DATA/out/`` and a human sends it. Agents do not
message anyone on Discord (AGENTS.md "Galaxus product rules" 7).

Only disagreements with a **human** label are reported. A ``model`` prelabel is not a label error: about a fifth of
the train cells are unchecked prelabels (FAQ.md, "Why do some train labels have the source ``model``?").

TODO(model): the prediction format is the model lane's to define; agreed shape is being settled on issue #7. Until
then :func:`out_of_fold_frame` accepts the long table below and the nested mapping, and nothing else:

    image_id, dimension, prediction, confidence
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from galaxus.data.classes import ID_COLUMN, dimensions
from galaxus.data.paths import out_dir

#: The columns the hosts expect, in this order. Do not add any.
REPORT_COLUMNS = (ID_COLUMN, "dimension", "label", "suggested_label")
DEFAULT_NAME = "label_errors.csv"
DEFAULT_THRESHOLD = 0.9

PREDICTION_COLUMNS = (ID_COLUMN, "dimension", "prediction", "confidence")

OutOfFold = pd.DataFrame | Mapping[str, Mapping[str, object]]


def out_of_fold_frame(predictions: OutOfFold) -> pd.DataFrame:
    """Out-of-fold predictions as a long table with the columns in :data:`PREDICTION_COLUMNS`.

    Accepts that table directly, or ``{image_id: {dimension: (class, confidence)}}`` where the value may also be a
    mapping with the keys ``prediction`` and ``confidence``.
    """
    if isinstance(predictions, pd.DataFrame):
        missing = [column for column in PREDICTION_COLUMNS if column not in predictions.columns]
        if missing:
            raise KeyError(f"the out-of-fold table is missing {missing}; expected columns {list(PREDICTION_COLUMNS)}")
        frame = predictions[list(PREDICTION_COLUMNS)].copy()
    else:
        rows = []
        for image_id, per_dimension in predictions.items():
            for dimension, value in (per_dimension or {}).items():
                if isinstance(value, Mapping):
                    prediction, confidence = value["prediction"], value["confidence"]
                else:
                    prediction, confidence = value
                rows.append((str(image_id), str(dimension), str(prediction), confidence))
        frame = pd.DataFrame(rows, columns=list(PREDICTION_COLUMNS))
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
