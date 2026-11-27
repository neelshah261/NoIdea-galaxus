"""Data lane: loading, image preparation, the official split and metric, submissions, label errors.

The competition data never lives in this repository (AGENTS.md "Galaxus product rules" 3). Every path comes from
the environment variable ``GALAXUS_DATA``; everything we generate goes to ``$GALAXUS_DATA/out/``.

The names below are re-exported lazily (PEP 562). Eagerly importing the submodules here would also preload
``galaxus.data.validate``, and ``python -m galaxus.data.validate`` would then execute that module a second time as
``__main__``, leaving two copies of ``SubmissionInvalid`` around.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checkers and editors only
    from galaxus.data.classes import ID_COLUMN, allowed_classes, dimensions
    from galaxus.data.images import opaque_rgb_on_white, prepare_image, prepared_jpeg, vision_tokens
    from galaxus.data.label_errors import find_label_errors, write_label_errors
    from galaxus.data.load import read_dev, read_final, read_part, read_sample_submission, read_train
    from galaxus.data.metric import evaluate, load_metric, score
    from galaxus.data.paths import GALAXUS_DATA_ENV, MissingDataDir, data_dir, data_file, out_dir
    from galaxus.data.split import scored_mask, solution_frame, split_holdout
    from galaxus.data.submission import write_submission
    from galaxus.data.validate import SubmissionInvalid, validate_submission

_EXPORTS = {
    "GALAXUS_DATA_ENV": "paths",
    "ID_COLUMN": "classes",
    "MissingDataDir": "paths",
    "SubmissionInvalid": "validate",
    "allowed_classes": "classes",
    "data_dir": "paths",
    "data_file": "paths",
    "dimensions": "classes",
    "evaluate": "metric",
    "find_label_errors": "label_errors",
    "load_metric": "metric",
    "opaque_rgb_on_white": "images",
    "out_dir": "paths",
    "prepare_image": "images",
    "prepared_jpeg": "images",
    "read_dev": "load",
    "read_final": "load",
    "read_part": "load",
    "read_sample_submission": "load",
    "read_train": "load",
    "score": "metric",
    "scored_mask": "split",
    "solution_frame": "split",
    "split_holdout": "split",
    "validate_submission": "validate",
    "vision_tokens": "images",
    "write_label_errors": "label_errors",
    "write_submission": "submission",
}

# Spelled out rather than ``sorted(_EXPORTS)`` so linters can see the re-exports in the
# ``TYPE_CHECKING`` block above are public. ``test_exports.py`` keeps the two in step.
__all__ = [
    "GALAXUS_DATA_ENV",
    "ID_COLUMN",
    "MissingDataDir",
    "SubmissionInvalid",
    "allowed_classes",
    "data_dir",
    "data_file",
    "dimensions",
    "evaluate",
    "find_label_errors",
    "load_metric",
    "opaque_rgb_on_white",
    "out_dir",
    "prepare_image",
    "prepared_jpeg",
    "read_dev",
    "read_final",
    "read_part",
    "read_sample_submission",
    "read_train",
    "score",
    "scored_mask",
    "solution_frame",
    "split_holdout",
    "validate_submission",
    "vision_tokens",
    "write_label_errors",
    "write_submission",
]


def __getattr__(name: str) -> object:
    """Import the submodule that owns ``name`` on first use."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return __all__
