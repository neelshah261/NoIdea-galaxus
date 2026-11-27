"""EXP-0006, mirror test-time augmentation for perspective (#48): classify the image and its mirror, then combine.

    uv run --env-file .env python -m galaxus.model.eval mirror-tta:zero-shot --exp-id EXP-0006
    ZERO_SHOT_PRESET=zero-shot-image-first python -m galaxus.serve.final_run --predictor galaxus.model.mirror_tta:from_env ...

- Only text-free images are mirrored (PLAN.md EXP-0006): the original answer's overlay dimension must be the
  no-overlay class, because mirrored text is not text. Every other image keeps the original answer.
- The mirror is the data lane's prepared JPEG (``cached_image(..., mirrored=True)``), handed to the inner predictor
  as an item whose id is ``<image_id>.mirror``. That is the name the cache already gives the mirror, so an inner
  predictor that reads through the cache at the same size gets the prepared mirror itself, not a JPEG of a JPEG.
- The mirrored answer is mapped back with ``galaxus.data.mirror.mirror_class`` (left and right swapped, a class
  without a partner unchanged), in the one dimension with left/right classes, found from the class list.
- Agreement keeps the answer. On disagreement the higher confidence wins when both answers carry one for that
  dimension, else the original. Every other dimension is the original image's answer.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from galaxus.data.classes import allowed_classes, no_overlay_class, overlay_dimension
from galaxus.data.images import IMAGE_SIZE, MIRROR_SUFFIX, cached_image
from galaxus.data.mirror import mirror_class, mirrorable_dimension
from galaxus.model.contract import ImageItem, Prediction, Predictor


class MirrorTTA:
    """Wraps any predictor: its answers, with the mirrorable dimension decided by the image and its mirror."""

    name = "mirror-tta"

    def __init__(self, inner: Predictor, allowed: Mapping[str, Iterable[str]] | None = None) -> None:
        self.inner = inner
        self.model = getattr(inner, "model", None)
        source = allowed if allowed is not None else allowed_classes()
        self.allowed = {dimension: frozenset(names) for dimension, names in source.items()}
        self.dimension = mirrorable_dimension(self.allowed)
        self.text_dimension = overlay_dimension()
        self.no_text_class = no_overlay_class()
        if self.no_text_class not in self.allowed.get(self.text_dimension, ()):
            raise ValueError(
                f"{self.no_text_class!r} is not a class of {self.text_dimension!r}, so no image would count as "
                "text-free and nothing would ever be mirrored"
            )
        self.image_size = getattr(inner, "image_size", IMAGE_SIZE)
        self._lock = threading.Lock()
        self.stats = {"images": 0, "skipped_text": 0, "agreements": 0, "disagreements": 0, "flips": 0}

    def _mirrored(self, item: ImageItem) -> ImageItem:
        path = cached_image(item.image_id, item.path, mirrored=True, image_size=self.image_size)
        return ImageItem(image_id=f"{item.image_id}{MIRROR_SUFFIX}", path=path, context=item.context)

    def _text_free(self, prediction: Prediction) -> bool:
        return prediction.labels.get(self.text_dimension) == self.no_text_class

    def _combine(self, original: Prediction, mirrored: Prediction) -> tuple[Prediction, tuple[str, ...]]:
        """The answer for one mirrored image, and the counters it raises."""
        back = mirror_class(self.dimension, mirrored.labels[self.dimension], self.allowed)
        if back == original.labels[self.dimension]:
            return original, ("agreements",)
        mine = (original.confidence or {}).get(self.dimension)
        theirs = (mirrored.confidence or {}).get(self.dimension)
        if mine is None or theirs is None or theirs <= mine:
            return original, ("disagreements",)
        flipped = replace(
            original,
            labels={**original.labels, self.dimension: back},
            confidence={**original.confidence, self.dimension: theirs},
        )
        return flipped, ("disagreements", "flips")

    def predict(self, batch: Sequence[ImageItem]) -> list[Prediction]:
        """The inner predictor's answers for ``batch``, after one more inner call on the text-free images' mirrors."""
        originals = self.inner.predict(batch)
        items = {item.image_id: item for item in batch}
        mirrors = [self._mirrored(items[p.image_id]) for p in originals if self._text_free(p)]
        mirrored = {p.image_id: p for p in self.inner.predict(mirrors)} if mirrors else {}
        answers, raised = [], ["images"] * len(originals)
        for original in originals:
            if not self._text_free(original):
                answers.append(original)
                raised.append("skipped_text")
                continue
            answer, counters = self._combine(original, mirrored[f"{original.image_id}{MIRROR_SUFFIX}"])
            answers.append(answer)
            raised.extend(counters)
        with self._lock:
            for counter in raised:
                self.stats[counter] += 1
        return answers

    def config(self) -> dict[str, Any]:
        """The inner predictor's config, plus what the EXP row records of the TTA: the dimension and the counts."""
        inner = getattr(self.inner, "config", None)
        with self._lock:
            stats = dict(self.stats)
        return {**(inner() if callable(inner) else {}), "mirror_tta": {"dimension": self.dimension, **stats}}


def from_env() -> MirrorTTA:
    """Mirror TTA around the zero-shot variant ``$ZERO_SHOT_PRESET`` names, for the runner, submit and final_run.

    Raises:
        ValueError: for a variant TTA cannot wrap (``zero_shot.mirror_tta_can_wrap``) or a name that is not one.
    """
    from galaxus.model import zero_shot

    name = zero_shot.preset_name()
    if not zero_shot.mirror_tta_can_wrap(name):
        raise ValueError(f"mirror TTA cannot wrap {name!r} yet: a crop variant's mirror would not be cropped the same way")
    return MirrorTTA(zero_shot.from_preset())
