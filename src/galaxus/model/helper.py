"""A helper image model picks the class, per dimension (Parth, after #172): class-balanced heads on an open image
encoder's embeddings. The encoder is SigLIP base by default: #112's encoder, fixed in #170.

* **The features:** each image's pooled embedding, from the data lane's per-image embedding cache
  (``galaxus.data.neighbours.embedding_dir``). An image not in it yet is embedded once and cached, so the
  similar-image index (#112) and this classifier share every embedding.
* **The classifier:** #51's heads (``galaxus.model.heads.fit_heads``): weighted cross-entropy for the rare classes,
  calibrated on product folds out of fold, fitted on the scored human labels of train minus the official holdout.
* **What it is for:** its EXP row on the official holdout, and its saved answers as an ``ensemble-agree`` member.
  The ensemble routes a dimension to it only where it beats the base on both product halves.
* **The rule it touches:** challenge_description.md:47 and FAQ.md:27 say Apertus performs the classification. In the
  submitted method, a dimension this helper answers needs the hosts' answer that a helper may pick the class
  recorded in ``decisions.md``. As a development measurement, any model is allowed (FAQ.md:30).

    uv run --extra local --env-file .env python -m galaxus.model.eval helper-siglip
    ENSEMBLE_MEMBERS="noctx=EXP-0025 ctx=EXP-0021 helper=<its EXP>" uv run --env-file .env \\
        python -m galaxus.model.eval ensemble-agree

Every image is embedded once, on CPU or GPU; train minus the holdout is about 9,400 images, already cached wherever
#112's index was built.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from galaxus.model import heads
from galaxus.types import ImageItem, Prediction

DEFAULT_ENCODER = "google/siglip-base-patch16-224"
FOLDS = 5
ID_COLUMN = "image_id"
FILE_COLUMN = "image_file"


def embeddings(
    pairs: Iterable[tuple[str, str]],
    encoder_name: str = DEFAULT_ENCODER,
    load_encoder: Callable[[str], Any] | None = None,
    directory: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """``{image_id: embedding}`` for ``(image_id, image file)`` pairs, from the cache; missing ones are embedded first."""
    from galaxus.data import neighbours

    pairs = [(str(image_id), str(path)) for image_id, path in pairs]
    folder = Path(directory) if directory is not None else neighbours.embedding_dir(encoder_name)
    missing = [(image_id, path) for image_id, path in pairs if not (folder / f"{image_id}.npy").is_file()]
    if missing:
        encoder = (load_encoder or neighbours.load_encoder)(encoder_name)
        neighbours.embed_rows(missing, encoder, encoder_name, directory=folder)
    return {image_id: np.load(folder / f"{image_id}.npy").astype(np.float32).reshape(-1) for image_id, _ in pairs}


def product_folds(seed: int = 0, n_folds: int = FOLDS) -> Callable[[pd.DataFrame], Iterable[tuple[pd.DataFrame, pd.DataFrame]]]:
    """Out-of-fold calibration splits, grouped by product (the data lane's ``split.product_folds``)."""
    from galaxus.data.split import product_folds as folds_of

    def folds(frame: pd.DataFrame) -> Iterable[tuple[pd.DataFrame, pd.DataFrame]]:
        fold = folds_of(frame, n_folds=n_folds, random_state=seed).to_numpy()
        for k in range(n_folds):
            yield frame[fold != k], frame[fold == k]

    return folds


class HelperPredictor(heads.HeadsPredictor):
    """#51's heads predictor on encoder embeddings: a batch's missing embeddings are computed in one call."""

    name = "helper"

    def __init__(self, model: heads.HeadsModel, encoder_name: str, embed: Callable[..., dict[str, np.ndarray]]) -> None:
        super().__init__(model)
        self.encoder_name = encoder_name
        self.embed = embed
        #: What the EXP row records: the helper model that picks the class, and it again as the helper.
        self.model = encoder_name
        self.model_revision = None
        self.helpers = (encoder_name,)

    def predict(self, batch: Sequence[ImageItem]) -> list[Prediction]:
        missing = [item for item in batch if str(item.image_id) not in self.features]
        if missing:
            self.features.update(self.embed([(str(item.image_id), str(item.path)) for item in missing], self.encoder_name))
        return super().predict(batch)


def factory(encoder_name: str = DEFAULT_ENCODER, embed: Callable[..., dict[str, np.ndarray]] = embeddings):
    """The eval script's factory: heads fitted on ``rows`` (train minus the official holdout), then the predictor."""

    def build(rows: pd.DataFrame, data: Any, seed: int) -> HelperPredictor:
        pairs = list(zip(rows[ID_COLUMN].astype(str), rows[FILE_COLUMN].astype(str), strict=True))
        features = embed(pairs, encoder_name)
        model = heads.fit_heads(rows, features, data.dimensions(), data.scored_mask, folds=product_folds(seed),
                                key=f"helper:{encoder_name}", seed=seed)
        return HelperPredictor(model, encoder_name, embed)

    return build


def from_env(environ: Mapping[str, str] | None = None) -> HelperPredictor:
    """For the runner, ``submit --run`` and ``final_run --predictor``: fitted on train minus the official holdout."""
    from galaxus.model.eval import DataLane

    data = DataLane.real()
    rows = data.split_holdout(data.read_train())[0]
    return factory()(rows, data, 0)
