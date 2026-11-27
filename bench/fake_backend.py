"""A CPU-only stand-in backend, so the speed matrix can be exercised where no GPU exists.

It is deliberately **not** a model. It does the per-image work every backend has to do anyway — decode, composite
transparency onto white, resize to the cell's image area, re-encode as JPEG — through :mod:`galaxus.data.images`,
counts the vision tokens that resize produces, and then answers one class per dimension.

Two rules it keeps:

- **No class name is written down here.** The dimensions and classes are read from :mod:`galaxus.data.classes` at
  runtime (AGENTS.md "Hard rules"). When no source for them is readable it answers a visibly fake placeholder and
  says so in ``labels_source``, so a timing run needs no competition data at all.
- **Its numbers are never product numbers.** A row measured with this backend proves the machinery, not Apertus.
  :mod:`bench.speed_matrix` marks such rows as machinery checks and keeps them out of the product table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image as PillowImage
from PIL import ImageOps

from galaxus.data import images

#: Backend name of this stand-in. It is not one of the product backends in :mod:`bench.speed_matrix`.
FAKE_BACKEND = "fake"
#: Precision of a backend that loads no weights at all.
NO_WEIGHTS = "none"
#: Shape of the images :func:`synthetic_items` writes: bigger than any cell, so every cell really resizes.
SYNTHETIC_SIZE = (1024, 768)
#: What a prediction says when no source for the real classes is readable. Not a class name, and never submitted.
PLACEHOLDER_DIMENSION = "_placeholder_dimension"
PLACEHOLDER_CLASS = "_placeholder_class"


@dataclass(frozen=True)
class FakePrediction:
    """The ``Prediction`` shape of PLAN.md "Lanes and contracts", as the runner checks it."""

    image_id: str
    labels: dict[str, str]
    confidence: dict[str, float] | None = None


class FakeBackendPredictor:
    """The ``Predictor`` contract on the CPU, with the image work of one grid cell actually done.

    Args:
        image_area: the cell's image area, passed to :func:`galaxus.data.images.prepare_image`.
        precision: must be :data:`NO_WEIGHTS`; this backend has no weights to quantize.
        prefix_caching: must be ``False``; this backend builds no prompt, so there is nothing to cache.
    """

    backend = FAKE_BACKEND

    def __init__(
        self,
        image_area: int = images.IMAGE_SIZE,
        precision: str = NO_WEIGHTS,
        prefix_caching: bool = False,
    ) -> None:
        if precision != NO_WEIGHTS:
            raise ValueError(f"the {FAKE_BACKEND} backend loads no weights, so precision must be {NO_WEIGHTS!r}")
        if prefix_caching:
            raise ValueError(f"the {FAKE_BACKEND} backend builds no prompt, so there is no prefix cache to turn on")
        self.image_area = int(image_area)
        self.precision = precision
        self.prefix_caching = prefix_caching
        self.images_seen = 0
        self.vision_tokens_total = 0
        self.batches: list[int] = []
        self._labels: dict[str, str] | None = None
        self.labels_source = "not resolved yet"

    @property
    def vision_tokens_per_image(self) -> float | None:
        """Mean vision tokens of the images this predictor prepared, or None before the first one.

        The count follows ``galaxus.data.images.PATCH_SIZE``, the reference notebook's patch size.
        TODO(model): Apertus' own patch size is measured in B6 (PLAN.md Q-0002); until then this is a proxy.
        """
        if not self.images_seen:
            return None
        return round(self.vision_tokens_total / self.images_seen, 2)

    def _resolve_labels(self) -> dict[str, str]:
        """One class per dimension, read from the data lane; a flagged placeholder when nothing is readable."""
        if self._labels is not None:
            return self._labels
        try:
            from galaxus.data.classes import allowed_classes

            classes = allowed_classes()
        # ClassesUnavailable and MissingDataDir are RuntimeErrors; a missing lane or file is the other two.
        except (ImportError, OSError, RuntimeError) as unavailable:
            self._labels = {PLACEHOLDER_DIMENSION: PLACEHOLDER_CLASS}
            self.labels_source = f"placeholder, no classes source ({type(unavailable).__name__}): timing only"
            return self._labels
        self._labels = {dimension: min(names) for dimension, names in classes.items()}
        self.labels_source = "galaxus.data.classes.allowed_classes()"
        return self._labels

    def predict(self, items: Any) -> list[FakePrediction]:
        """Prepare every image of the batch the way a backend would, then answer the resolved labels."""
        batch = list(items)
        self.batches.append(len(batch))
        labels = self._resolve_labels()
        predictions = []
        for item in batch:
            prepared = images.prepare_image(item.path, image_size=self.image_area)
            self.vision_tokens_total += images.vision_tokens(prepared)
            self.images_seen += 1
            predictions.append(FakePrediction(str(item.image_id), dict(labels)))
        return predictions


def synthetic_image(index: int) -> PillowImage.Image:
    """A deterministic synthetic RGBA image: gradients plus transparency, so compositing and resizing both happen.

    Synthetic pixels only, built in this process. Never a competition image (AGENTS.md "Work → PR" 5).
    """
    width, height = SYNTHETIC_SIZE
    linear = PillowImage.linear_gradient("L").rotate(index * 11 % 360).resize((width, height))
    radial = PillowImage.radial_gradient("L").resize((width, height))
    alpha = radial.point(lambda value: 255 if value > (index * 17) % 200 else 0)
    return PillowImage.merge("RGBA", (linear, radial, ImageOps.mirror(linear), alpha))


def synthetic_items(count: int, directory: Any) -> list[Any]:
    """``count`` synthetic images written into ``directory``, as the item type the runner uses."""
    from pathlib import Path

    from galaxus.serve.items import image_item_type

    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    item = image_item_type()
    built = []
    for index in range(count):
        path = folder / f"synthetic_{index:04d}.png"
        if not path.exists():
            synthetic_image(index).save(path)
        built.append(item(image_id=f"synthetic_{index:04d}", path=str(path), context=None))
    return built
