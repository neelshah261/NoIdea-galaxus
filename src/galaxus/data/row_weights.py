"""Per-row training weights from two independent signals (#136): label noise down, hard examples up.

The hosts' never-tried idea (#17) is "hard-example mining and label-noise handling". Detection already exists in
this lane - out-of-fold model disagreement (#7) and neighbour disagreement (#114) - and this module turns it into
something training can consume.

The reason two signals are needed is that **a model disagreeing with a human label means two opposite things**,
and one signal cannot tell them apart:

* the neighbours side with the model, so two independent sources dispute the label -> probably a **label error**,
  and the row should be down-weighted, dropped, or relabelled;
* the neighbours side with the human, so the label is corroborated and the model is simply wrong -> a **hard
  example**, and the row should be oversampled.

Treating every disagreement as noise would throw away exactly the rows worth training on, and treating every
disagreement as a hard example would train on mistakes. Where the neighbours have no opinion the cell is left
alone: one signal is not evidence here.

The output never enters git. It holds competition labels, so it falls under AGENTS.md product rule 3 and is
written to ``$GALAXUS_DATA/out/`` behind the same work-tree refusal as the label-error csv.

Consumers: ``galaxus.model.lora.build_training_set`` and #51's heads (model lane - this module only writes the
file; the loader option is requested on #136, not written here).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from galaxus.data.classes import dimensions
from galaxus.data.label_errors import ID_COLUMN, _checked_target, out_of_fold_frame
from galaxus.data.split import scored_mask

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

DEFAULT_NAME = "row_weights.csv"
WEIGHT_COLUMNS = (ID_COLUMN, "dimension", "weight", "reason")
RELABEL_COLUMNS = (ID_COLUMN, "dimension", "label", "suggested_label")

#: A cell nothing is known against keeps this weight, so a weight file is complete rather than a patch.
DEFAULT_WEIGHT = 1.0
#: Both signals dispute the label: train on it, but quietly.
SUSPECT_WEIGHT = 0.25
#: Both signals dispute it and both are very confident: do not train on it at all.
DROP_WEIGHT = 0.0
#: The label is corroborated and the model still got it wrong: the row worth seeing twice.
HARD_WEIGHT = 2.0

#: A model prediction below this is not evidence of anything.
DEFAULT_MODEL_THRESHOLD = 0.90
#: Neighbour agreement, which at k=3 can only be 1.00, 0.67 or 0.33 - so 0.67 means "two of three".
DEFAULT_NEIGHBOUR_THRESHOLD = 0.67
#: Above this on both signals, a suspect cell is dropped rather than down-weighted.
DEFAULT_DROP_THRESHOLD = 0.98

SUSPECT_REASON = "suspect_label"
DROP_REASON = "suspect_label_dropped"
HARD_REASON = "hard_example"


def _signal(records: Iterable[object] | str | os.PathLike[str], name: str) -> pd.DataFrame:
    """One signal as ``image_id, dimension, <name>, <name>_confidence``, read the way #7 reads predictions."""
    frame = out_of_fold_frame(records)
    frame = frame.rename(columns={"prediction": name, "confidence": f"{name}_confidence"})
    return frame[[ID_COLUMN, "dimension", name, f"{name}_confidence"]]


def _human_cells(train: pd.DataFrame, dimension_names: tuple[str, ...]) -> pd.DataFrame:
    """The scored human labels of ``train``, long: one row per (image, dimension) cell that counts.

    Only scored cells are used. A model prelabel is the thing being checked, and a cell the labeling rules call
    irrelevant is not trained on either, so neither belongs in a weight file.
    """
    rows: list[dict[str, Any]] = []
    for dimension in dimension_names:
        keep = scored_mask(train, dimension) & train[dimension].notna()
        for image_id, label in zip(train.loc[keep, ID_COLUMN], train.loc[keep, dimension], strict=True):
            rows.append({ID_COLUMN: str(image_id), "dimension": dimension, "label": str(label)})
    return pd.DataFrame(rows, columns=[ID_COLUMN, "dimension", "label"])


def judge_cells(
    train: pd.DataFrame,
    model_predictions: Iterable[object] | str | os.PathLike[str],
    neighbour_votes: Iterable[object] | str | os.PathLike[str],
    dimension_names: tuple[str, ...] | None = None,
    model_threshold: float = DEFAULT_MODEL_THRESHOLD,
    neighbour_threshold: float = DEFAULT_NEIGHBOUR_THRESHOLD,
    drop_threshold: float = DEFAULT_DROP_THRESHOLD,
) -> pd.DataFrame:
    """Every scored human cell, with what each signal said and the verdict the two of them support.

    Returns a frame with the human ``label``, each signal's prediction and confidence, and a ``verdict`` of
    ``""`` (nothing to say), :data:`SUSPECT_REASON`, :data:`DROP_REASON` or :data:`HARD_REASON`.
    """
    names = tuple(dimension_names) if dimension_names is not None else dimensions()
    cells = _human_cells(train, names)
    if cells.empty:
        raise ValueError("no scored human cells in this train table, so there is nothing to weight")

    model = _signal(model_predictions, "model")
    neighbours = _signal(neighbour_votes, "neighbour")
    merged = cells.merge(model, on=[ID_COLUMN, "dimension"], how="left").merge(
        neighbours, on=[ID_COLUMN, "dimension"], how="left"
    )

    model_says = merged["model"].notna() & (merged["model"] != merged["label"])
    model_sure = merged["model_confidence"].fillna(0.0).astype(float) >= model_threshold
    neighbour_sure = merged["neighbour_confidence"].fillna(0.0).astype(float) >= neighbour_threshold
    neighbour_disputes = merged["neighbour"].notna() & (merged["neighbour"] != merged["label"]) & neighbour_sure
    neighbour_confirms = merged["neighbour"].notna() & (merged["neighbour"] == merged["label"]) & neighbour_sure

    disputed = model_says & model_sure & neighbour_disputes
    # Both signals must name the SAME replacement before the cell counts as a label error: the model saying
    # "not this" and the neighbours saying "not that either" is two kinds of doubt, not one finding.
    agreed = disputed & (merged["model"] == merged["neighbour"])
    both_certain = (
        (merged["model_confidence"].fillna(0.0).astype(float) >= drop_threshold)
        & neighbour_sure
        & (merged["neighbour_confidence"].fillna(0.0).astype(float) >= drop_threshold)
    )

    merged["verdict"] = ""
    merged.loc[agreed, "verdict"] = SUSPECT_REASON
    merged.loc[agreed & both_certain, "verdict"] = DROP_REASON
    merged.loc[model_says & model_sure & neighbour_confirms, "verdict"] = HARD_REASON
    return merged


def row_weights(judged: pd.DataFrame) -> pd.DataFrame:
    """``judge_cells`` output as the weight file: one row per cell, ``weight`` and ``reason``."""
    weights = {SUSPECT_REASON: SUSPECT_WEIGHT, DROP_REASON: DROP_WEIGHT, HARD_REASON: HARD_WEIGHT}
    frame = judged[[ID_COLUMN, "dimension"]].copy()
    frame["weight"] = judged["verdict"].map(weights).fillna(DEFAULT_WEIGHT).astype(float)
    frame["reason"] = judged["verdict"]
    return frame[list(WEIGHT_COLUMNS)].reset_index(drop=True)


def suggested_relabels(judged: pd.DataFrame) -> pd.DataFrame:
    """The bonus csv (#7, #136): the cells both signals dispute, and what they say instead.

    Only cells where the two signals agree on a replacement appear, which is what makes a suggestion worth
    sending. Ordered by the model's confidence so the strongest candidates read first.
    """
    disputed = judged[judged["verdict"].isin((SUSPECT_REASON, DROP_REASON))].copy()
    disputed = disputed.sort_values("model_confidence", ascending=False)
    disputed["suggested_label"] = disputed["model"]
    return disputed[list(RELABEL_COLUMNS)].reset_index(drop=True)


def per_image_weights(weights: pd.DataFrame, how: str = "min") -> pd.DataFrame:
    """Collapse per-cell weights to one weight per image, for a consumer that trains on whole images.

    The LoRA's training row is an image with all six dimensions in its target, so it cannot apply a weight per
    cell; #51's heads are per dimension and should use the per-cell file directly.

    ``how="min"`` is the cautious collapse: an image with one disputed cell is treated as disputed. ``how="mean"``
    dilutes a single bad cell across the row. There is no correct answer here, so the choice is the caller's and
    the file keeps both possible.
    """
    if how not in ("min", "mean"):
        raise ValueError(f"how must be 'min' or 'mean', not {how!r}")
    grouped = weights.groupby(ID_COLUMN, sort=True)["weight"]
    collapsed = (grouped.min() if how == "min" else grouped.mean()).reset_index()
    return collapsed


def write_row_weights(
    weights: pd.DataFrame, path: str | os.PathLike[str] | None = None, name: str = DEFAULT_NAME
) -> Path:
    """Write the weight file to ``$GALAXUS_DATA/out/`` and return the path. It never enters git.

    The guard is in this function and not only in the command, because #51 and the LoRA may call it directly and
    a caller passing its own path must be refused too: these rows carry competition labels (product rule 3).
    """
    columns = list(WEIGHT_COLUMNS)
    if list(weights.columns) != columns:
        raise ValueError(f"the weight file must have exactly the columns {columns}, got {list(weights.columns)}")
    target = _checked_target(path, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    weights.to_csv(target, index=False)
    return target


def _summarise(weights: pd.DataFrame) -> str:
    counts = weights["reason"].replace("", "unchanged").value_counts()
    return ", ".join(f"{reason}: {count:,}" for reason, count in counts.items())


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m galaxus.data.row_weights``: the weight file from the two signals (#136).

    Prints counts and class names only - never an image id (product rule 3).
    """
    import argparse

    from galaxus.data.load import read_train

    parser = argparse.ArgumentParser(
        prog="python -m galaxus.data.row_weights",
        description="Per-row training weights: suspected label errors down, hard examples up (#136).",
    )
    parser.add_argument("--model-predictions", required=True, help="out-of-fold predictions jsonl, or its folder")
    parser.add_argument("--neighbour-votes", required=True, help="neighbour votes jsonl (galaxus.data.neighbours)")
    parser.add_argument("--dimensions", help="comma-separated subset of dimensions (default: all of them)")
    parser.add_argument("--model-threshold", type=float, default=DEFAULT_MODEL_THRESHOLD)
    parser.add_argument("--neighbour-threshold", type=float, default=DEFAULT_NEIGHBOUR_THRESHOLD)
    parser.add_argument("--drop-threshold", type=float, default=DEFAULT_DROP_THRESHOLD)
    parser.add_argument("--per-image", choices=("min", "mean"), help="also write one weight per image this way")
    parser.add_argument("--dry-run", action="store_true", help="print the counts and write nothing")
    arguments = parser.parse_args(argv)

    names = tuple(name.strip() for name in arguments.dimensions.split(",")) if arguments.dimensions else None
    try:
        judged = judge_cells(
            read_train(),
            arguments.model_predictions,
            arguments.neighbour_votes,
            dimension_names=names,
            model_threshold=arguments.model_threshold,
            neighbour_threshold=arguments.neighbour_threshold,
            drop_threshold=arguments.drop_threshold,
        )
    except (OSError, KeyError, ValueError) as failure:
        print(f"cannot work out the row weights: {failure}", file=sys.stderr)
        return 1

    weights = row_weights(judged)
    relabels = suggested_relabels(judged)
    print(f"[row-weights] {len(weights):,} scored cell(s): {_summarise(weights)}")
    print(f"[row-weights] {len(relabels):,} cell(s) both signals dispute, for the bonus csv (#7)")
    agreed = judged["verdict"].isin((SUSPECT_REASON, DROP_REASON))
    disagreed_one_way = (judged["model"].notna() & (judged["model"] != judged["label"])).sum()
    print(
        f"[row-weights] the model disputed {disagreed_one_way:,} cell(s); the neighbours backed it on "
        f"{int(agreed.sum()):,} of them, which is why one signal is not enough"
    )
    if arguments.dry_run:
        print("[row-weights] --dry-run: nothing written")
        return 0

    written = write_row_weights(weights)
    print(f"[row-weights] -> {written}")
    if arguments.per_image:
        collapsed = per_image_weights(weights, how=arguments.per_image)
        frame = collapsed.assign(dimension="", reason="")[list(WEIGHT_COLUMNS)]
        print(f"[row-weights] -> {write_row_weights(frame, name='row_weights_per_image.csv')}")
    print("[row-weights] holds train labels: never commit it (product rule 3).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
