"""From a competition table to the ``ImageItem`` list a ``Predictor`` gets, and loading a predictor by name.

The contract is ``galaxus.types.ImageItem{image_id, path, context | None}`` (design, #11 / #80; the format is in
``contracts/prediction.md``).
"""

from __future__ import annotations

import importlib
from typing import Any

import pandas as pd

ID_COLUMN = "image_id"
IMAGE_FILE_COLUMN = "image_file"
#: Columns that are never product context: identifiers and paths. Label columns are found by their ``_source``.
NOT_CONTEXT = frozenset({"image_id", "image_path", "image_file", "product_id"})
LABEL_METADATA_SUFFIXES = ("_source", "_relevant")


def image_item_type() -> type:
    """The shared ``galaxus.types.ImageItem`` (#82: the serve lane's stand-in is gone)."""
    from galaxus.types import ImageItem

    return ImageItem


def context_columns(frame: pd.DataFrame) -> list[str]:
    """The product-context columns of a competition table.

    Everything except ids, paths, labels and label metadata. A label column is one that has a
    ``<column>_source`` sibling, so no dimension or class name is written down here.
    """
    columns = [str(column) for column in frame.columns]
    labels = {column for column in columns if f"{column}_source" in columns}
    return [
        column
        for column in columns
        if column not in NOT_CONTEXT
        and column not in labels
        and not column.endswith(LABEL_METADATA_SUFFIXES)
    ]


def items_from_frame(frame: pd.DataFrame, limit: int | None = None) -> list[Any]:
    """One item per row of a table read with ``galaxus.data.load.read_part``, in the table's order.

    ``context`` is the product context of the row, or ``None`` when every context cell is empty: in production the
    context can be missing (challenge_description.md, "The task"), and a predictor must cope with that.
    """
    missing = [column for column in (ID_COLUMN, IMAGE_FILE_COLUMN) if column not in frame.columns]
    if missing:
        raise KeyError(f"{missing} missing: read the table with galaxus.data.load.read_part, which adds image_file")
    rows = frame if limit is None else frame.head(limit)
    item = image_item_type()
    columns = context_columns(rows)
    items = []
    for record in rows.to_dict(orient="records"):
        context = {column: str(record[column]) for column in columns if str(record[column]).strip()}
        items.append(
            item(image_id=str(record[ID_COLUMN]), path=str(record[IMAGE_FILE_COLUMN]), context=context or None)
        )
    return items


def load_predictor(spec: str) -> Any:
    """A predictor from ``"package.module:name"``.

    ``name`` may be a class or a factory function (called with no arguments) or an already-built object with a
    ``predict`` method. Example: ``galaxus.model.baselines:MajorityPredictor`` (model lane, issue #12).
    """
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"predictor must look like 'package.module:name', got {spec!r}")
    target = getattr(importlib.import_module(module_name), attribute)
    if isinstance(target, type) or (callable(target) and not hasattr(target, "predict")):
        predictor = target()
    else:
        predictor = target
    if not callable(getattr(predictor, "predict", None)):
        raise TypeError(f"{spec} did not give an object with a predict(items) method")
    return predictor
