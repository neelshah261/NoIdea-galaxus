"""The prediction contract the model lane codes against: ``ImageItem`` → ``Predictor.predict`` → ``Prediction``.

Design owns it. PLAN.md "Lanes and contracts" (branch ``design/3-plan``) defines the names and fields, and B0
(issue #11) puts them in ``galaxus.types``. Until #11 is on main, this module carries a copy with exactly those
names and fields so the model lane can start; once ``galaxus.types`` exists, its classes are used instead and every
test runs against them.

TODO(model): delete the copy once #11 is merged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

try:
    from galaxus.types import ImageItem, Prediction, Predictor
except ModuleNotFoundError as missing:
    if missing.name != "galaxus.types":
        raise

    @dataclass(frozen=True)
    class ImageItem:
        """One image to classify. ``context`` is its product context from the catalog, or None when it is missing."""

        image_id: str
        path: str | Path
        context: Mapping[str, str] | None = None

    @dataclass(frozen=True)
    class Prediction:
        """One class per dimension for one image: never an empty or an unknown class (PLAN.md, prediction.md)."""

        image_id: str
        labels: Mapping[str, str]
        confidence: Mapping[str, float] | None = None

    @runtime_checkable
    class Predictor(Protocol):
        """Anything that classifies a batch of images: majority, zero-shot, heads, LoRA, ensembles."""

        def predict(self, batch: Sequence[ImageItem]) -> list[Prediction]: ...


__all__ = ["ImageItem", "Prediction", "Predictor"]
