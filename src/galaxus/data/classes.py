"""The six dimensions and the class names allowed in each.

The dimensions and their classes are defined by the challenge, not by us: see
[challenge_description.md, "Classes"](../../../challenge_description.md#classes) and ``labeling_guideline.md`` in the
Kaggle data. AGENTS.md "Hard rules" forbids restating them from memory, so nothing here hard-codes a class name;
every name is read from a source file at runtime, in this order:

1. ``contracts/classes.json`` in the repository, owned by design.
   TODO(design): add it (issue #6). Expected shape: ``{"<dimension>": ["<class>", ...], ...}``, dimensions in
   submission-column order.
2. ``metric.py`` from ``$GALAXUS_DATA`` (its ``DIMENSIONS`` and ``VALID_CLASSES``), the code that scores us.
3. the header of ``sample_submission.csv`` plus the class names that occur in it and in ``train.csv``.

Source 3 can only see classes that occur in those files, so it is a floor, not the truth: prefer 1 or 2 before
rejecting a prediction.
"""

from __future__ import annotations

import json
from pathlib import Path

ID_COLUMN = "image_id"

CONTRACTS_CLASSES = Path(__file__).resolve().parents[3] / "contracts" / "classes.json"


class ClassesUnavailable(RuntimeError):
    """No source for the dimensions or their classes could be read."""


def _from_contract() -> dict[str, list[str]] | None:
    """The contract's classes per dimension, or None when design has not added the file yet."""
    if not CONTRACTS_CLASSES.is_file():
        return None
    contract = json.loads(CONTRACTS_CLASSES.read_text(encoding="utf-8"))
    # TODO(design): pin this shape in contracts/classes.json; both layouts are accepted until then.
    mapping = contract.get("dimensions", contract) if isinstance(contract, dict) else contract
    if not isinstance(mapping, dict) or not mapping:
        raise ClassesUnavailable(f"{CONTRACTS_CLASSES} does not map dimensions to class names")
    return {str(dimension): [str(name) for name in names] for dimension, names in mapping.items()}


def _from_metric() -> dict[str, list[str]] | None:
    """The scoring code's classes per dimension, or None when ``metric.py`` cannot be loaded."""
    from galaxus.data.metric import load_metric

    try:
        metric = load_metric()
    except (ImportError, OSError, RuntimeError):
        # No metric.py, no $GALAXUS_DATA, or a metric.py that will not import: fall through to the next source.
        return None
    valid = getattr(metric, "VALID_CLASSES", None)
    order = getattr(metric, "DIMENSIONS", None)
    if not valid or not order:
        return None
    return {str(dimension): [str(name) for name in valid[dimension]] for dimension in order}


def _from_files() -> dict[str, list[str]] | None:
    """The classes that occur in ``sample_submission.csv`` and ``train.csv``, or None when neither can be read."""
    from galaxus.data.load import read_sample_submission, read_train

    try:
        sample = read_sample_submission()
    except (OSError, RuntimeError):
        return None
    found: dict[str, set[str]] = {
        str(column): {str(value) for value in sample[column].unique() if str(value)}
        for column in sample.columns
        if column != ID_COLUMN
    }
    try:
        train = read_train()
    except (OSError, RuntimeError):
        train = None
    if train is not None:
        for dimension in found:
            if dimension in train.columns:
                found[dimension] |= {str(value) for value in train[dimension].unique() if str(value)}
    return {dimension: sorted(names) for dimension, names in found.items()}


def _resolved() -> dict[str, list[str]]:
    for source in (_from_contract, _from_metric, _from_files):
        resolved = source()
        if resolved:
            return resolved
    raise ClassesUnavailable(
        "no source for the classes: add contracts/classes.json, or set $GALAXUS_DATA so that metric.py and "
        "sample_submission.csv can be read"
    )


def _roles() -> dict[str, str]:
    """The contract's ``roles`` block: the names whose *meaning* the code needs, not just their spelling."""
    if not CONTRACTS_CLASSES.is_file():
        raise ClassesUnavailable(
            f"{CONTRACTS_CLASSES} is not there, so which dimension carries the text overlay, and which of its "
            "classes means 'no text', are unknown. They are design's to publish (contracts/classes.json, "
            '"roles"); pass them explicitly in the meantime.'
        )
    contract = json.loads(CONTRACTS_CLASSES.read_text(encoding="utf-8"))
    block = contract.get("roles") if isinstance(contract, dict) else None
    if not isinstance(block, dict):
        raise ClassesUnavailable(f'{CONTRACTS_CLASSES} has no "roles" block naming the dimensions with a meaning')
    return {str(name): str(value) for name, value in block.items() if isinstance(value, str)}


def _role(name: str) -> str:
    value = _roles().get(name)
    if not value:
        raise ClassesUnavailable(f'{CONTRACTS_CLASSES} "roles" does not name {name}')
    return value


def overlay_dimension() -> str:
    """The dimension that says whether the image carries a text overlay.

    From ``contracts/classes.json`` "roles", which quotes challenge_description.md "Classes". Never typed in lane
    code (AGENTS.md "Hard rules").
    """
    return _role("overlay_dimension")


def no_overlay_class() -> str:
    """The class of :func:`overlay_dimension` meaning "this image carries no text".

    A real class, scored like any other, per the quote the contract carries.
    """
    return _role("no_overlay_class")


def dimensions() -> tuple[str, ...]:
    """The six dimensions, in the column order of the submission file."""
    return tuple(_resolved())


def allowed_classes() -> dict[str, frozenset[str]]:
    """The class names allowed in each dimension."""
    return {dimension: frozenset(names) for dimension, names in _resolved().items()}
