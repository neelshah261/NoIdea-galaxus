"""The shared contract every lane codes against: ``ImageItem`` -> ``Predictor.predict`` -> ``Prediction``.

Design owns this module (B0, issue #11); the formats are written out in ``contracts/prediction.md``. It has the same
names and fields as the stand-ins the lanes carried until now (``galaxus.model.contract`` and
``galaxus.serve.items.PendingImageItem``), and both switch to these classes on their own once this module exists.

Nothing here names a dimension or a class: those come from ``contracts/classes.json`` through
``galaxus.data.classes`` (``dimensions()``, ``allowed_classes()``), never typed in code (AGENTS.md "Hard rules").
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ImageItem:
    """One image to classify.

    ``path`` is the image file. ``context`` is the product context from the catalog (title, type, brand, category,
    ...), or ``None`` when it is missing: in production it can be missing or poor (challenge_description.md,
    "The task"), so a predictor must cope without it.
    """

    image_id: str
    path: str | Path
    context: Mapping[str, str] | None = None


@dataclass(frozen=True)
class Prediction:
    """One class per dimension for one image.

    ``labels`` maps every dimension to one allowed class: never an empty cell and never an unknown class
    (``contracts/prediction.md``; the data lane's validator enforces it before any file goes to Kaggle).
    ``confidence`` maps dimensions to a probability in [0, 1] for the chosen class, or is ``None`` when the method
    gives none.
    """

    image_id: str
    labels: Mapping[str, str]
    confidence: Mapping[str, float] | None = None


@runtime_checkable
class Predictor(Protocol):
    """Anything that classifies a batch of images: majority, zero-shot, heads, LoRA, ensembles.

    ``predict`` returns one ``Prediction`` per item, for the same image ids. The runner, the eval script, the
    benchmark and the Saturday script only ever call this method, so a new method changes nothing else.
    """

    def predict(self, batch: Sequence[ImageItem]) -> list[Prediction]: ...


__all__ = ["ImageItem", "Prediction", "Predictor"]
