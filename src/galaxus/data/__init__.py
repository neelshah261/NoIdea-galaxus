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
    from galaxus.data.images import (
        cached_jpeg,
        fill_image_cache,
        image_cache_dir,
        opaque_rgb_on_white,
        prepare_image,
        prepared_jpeg,
        vision_tokens,
    )
    from galaxus.data.label_errors import find_label_errors, write_label_errors
    from galaxus.data.load import read_dev, read_final, read_part, read_sample_submission, read_train
    from galaxus.data.metric import evaluate, load_metric, score
    from galaxus.data.paths import GALAXUS_DATA_ENV, MissingDataDir, cache_dir, data_dir, data_file, out_dir
    from galaxus.data.split import iter_product_folds, product_folds, scored_mask, solution_frame, split_holdout
    from galaxus.data.submission import write_submission
    from galaxus.data.validate import SubmissionInvalid, validate_submission

_EXPORTS = {
    "GALAXUS_DATA_ENV": "paths",
    "ID_COLUMN": "classes",
    "MissingDataDir": "paths",
    "SubmissionInvalid": "validate",
    "allowed_classes": "classes",
    "cache_dir": "paths",
    "cached_jpeg": "images",
    "data_dir": "paths",
    "data_file": "paths",
    "dimensions": "classes",
    "evaluate": "metric",
    "fill_image_cache": "images",
    "find_label_errors": "label_errors",
    "iter_product_folds": "split",
    "image_cache_dir": "images",
    "load_metric": "metric",
    "opaque_rgb_on_white": "images",
    "out_dir": "paths",
    "prepare_image": "images",
    "prepared_jpeg": "images",
    "product_folds": "split",
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
    "cache_dir",
    "cached_jpeg",
    "data_dir",
    "data_file",
    "dimensions",
    "evaluate",
    "fill_image_cache",
    "find_label_errors",
    "image_cache_dir",
    "iter_product_folds",
    "load_metric",
    "opaque_rgb_on_white",
    "out_dir",
    "prepare_image",
    "prepared_jpeg",
    "product_folds",
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
