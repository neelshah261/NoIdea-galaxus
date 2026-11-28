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

import io
import os
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from galaxus.model import heads
from galaxus.model.calibration import apply_bias, fit_bias
from galaxus.types import ImageItem, Prediction

DEFAULT_ENCODER = "google/siglip-base-patch16-224"
#: Every encoder's Hugging Face revision, pinned, so a new upload to its repository can never change the embeddings
#: under a run (determinism, #182). SigLIP base's is the one #170 verified from the HF API; every cached embedding
#: was made from it. DINOv2 base and SigLIP 2 base are the snapshots EXP-0059's cached embeddings were made from
#: (Neel's laptop, #182). An encoder that is not listed here is refused rather than loaded at its newest upload.
REVISIONS = {
    DEFAULT_ENCODER: "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
    "facebook/dinov2-base": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
    "google/siglip2-base-patch16-224": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
}
#: EXP-0059's encoders, **in this order**: each image's three embeddings, each divided by its own L2 norm, are
#: concatenated in this order before the heads (#182). Changing the order changes the features.
FUSED_ENCODERS = ("google/siglip-base-patch16-224", "facebook/dinov2-base", "google/siglip2-base-patch16-224")
FUSED_NAME = "fused:" + "+".join(FUSED_ENCODERS)
#: The encoders behind a helper name: one for a single encoder, three for the fused helper.
COMPONENTS = {FUSED_NAME: FUSED_ENCODERS}
#: ``HELPER_FIT_ALL=1``: the dev and final heads are fitted on every labelled row, the official holdout included
#: (#176). EXP rows never use it: the eval script fits on train minus the holdout and scores the holdout.
FIT_ALL_ENV = "HELPER_FIT_ALL"
#: What the pinned snapshot needs: the processor's and the model's configs and the safetensors weights (the repo
#: also holds pytorch_model.bin, the same weights again, and a text tokenizer the image side never uses).
SNAPSHOT_FILES = ["config.json", "preprocessor_config.json", "model.safetensors"]
FOLDS = 5
ID_COLUMN = "image_id"
FILE_COLUMN = "image_file"


def pinned_revision(encoder_name: str) -> str:
    """The pinned revision of ``encoder_name``, or a ValueError that says where to pin one."""
    revision = REVISIONS.get(encoder_name)
    if revision is None:
        raise ValueError(f"{encoder_name} has no pinned revision: add it to galaxus.model.helper.REVISIONS (the commit "
                         "sha from the Hugging Face API), so the embeddings cannot change under a run")
    return revision


def load_pinned_encoder(encoder_name: str) -> Any:
    """The data lane's encoder (``neighbours.load_encoder``), loaded from the pinned snapshot of ``encoder_name``."""
    from huggingface_hub import snapshot_download

    from galaxus.data import neighbours

    snapshot = snapshot_download(repo_id=encoder_name, revision=pinned_revision(encoder_name),
                                 allow_patterns=SNAPSHOT_FILES)
    return neighbours.load_encoder(snapshot)


_ENCODERS: dict[str, Any] = {}
_LOCKS: dict[str, threading.Lock] = {}
_GUARD = threading.Lock()


def _lock_for(encoder_name: str) -> threading.Lock:
    with _GUARD:
        return _LOCKS.setdefault(encoder_name, threading.Lock())


def shared_encoder(encoder_name: str) -> Any:
    """One loaded encoder per name for the whole process (#181, found by Swarup's drill: the runner's batches each
    reloaded SigLIP, a Hub check plus a model load per batch, about 330 times for the final images)."""
    with _lock_for(f"load:{encoder_name}"):
        if encoder_name not in _ENCODERS:
            _ENCODERS[encoder_name] = load_pinned_encoder(encoder_name)
        return _ENCODERS[encoder_name]


def embeddings(
    pairs: Iterable[tuple[str, str]],
    encoder_name: str = DEFAULT_ENCODER,
    load_encoder: Callable[[str], Any] | None = None,
    directory: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """``{image_id: embedding}`` for ``(image_id, image file)`` pairs, from the cache; missing ones are embedded first.

    The cache is keyed by the encoder's name, as the data lane's index keeps it; the encoder itself is loaded once per
    process at its pinned revision (:func:`shared_encoder`). One encoder's embedding runs one at a time, so two
    ensemble members that share an encoder never write the same cache file at once: the second finds it cached.
    """
    from galaxus.data import neighbours

    pairs = [(str(image_id), str(path)) for image_id, path in pairs]
    folder = Path(directory) if directory is not None else neighbours.embedding_dir(encoder_name)
    with _lock_for(f"embed:{folder}"):
        missing = [(image_id, path) for image_id, path in pairs if not (folder / f"{image_id}.npy").is_file()]
        if missing:
            encoder = (load_encoder or shared_encoder)(encoder_name)
            neighbours.embed_rows(missing, encoder, encoder_name, directory=folder)
        return {image_id: np.load(folder / f"{image_id}.npy").astype(np.float32).reshape(-1) for image_id, _ in pairs}


def unit_length(vector: np.ndarray) -> np.ndarray:
    """``vector`` divided by its own L2 norm (an all-zero vector stays as it is)."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def fused(per_encoder: Callable[..., dict[str, np.ndarray]],
          encoders: Sequence[str] = FUSED_ENCODERS) -> Callable[..., dict[str, np.ndarray]]:
    """EXP-0059's features from a per-encoder embedding function: per image, each encoder's vector at unit length,
    concatenated in ``encoders``' order (#182)."""

    def embed(pairs: Iterable[tuple[str, str]], _name: str = FUSED_NAME) -> dict[str, np.ndarray]:
        pairs = [(str(image_id), str(path)) for image_id, path in pairs]
        parts = [per_encoder(pairs, name) for name in encoders]
        return {image_id: np.concatenate([unit_length(part[image_id]) for part in parts]) for image_id, _ in pairs}

    return embed


def fused_embeddings(pairs: Iterable[tuple[str, str]], _name: str = FUSED_NAME) -> dict[str, np.ndarray]:
    """``helper-fused``'s features (EXP-0059): SigLIP base, DINOv2 base and SigLIP 2 base, each at unit length."""
    return fused(embeddings)(pairs, _name)


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
        #: What the EXP row records: the helper model that picks the class, its encoders and their pinned revisions.
        components = COMPONENTS.get(encoder_name, (encoder_name,))
        self.model = encoder_name
        revisions = [REVISIONS.get(name) for name in components]
        self.model_revision = ", ".join(str(r) for r in revisions) if len(revisions) > 1 else revisions[0]
        self.helpers = tuple(components)

    def predict(self, batch: Sequence[ImageItem]) -> list[Prediction]:
        missing = [item for item in batch if str(item.image_id) not in self.features]
        if missing:
            self.features.update(self.embed([(str(item.image_id), str(item.path)) for item in missing], self.encoder_name))
        return super().predict(batch)


# ---------------------------------------------------------------- the mirror-aware perspective head (#176, #182)
#
# A mirror changes one thing in a photo: left and right. So the perspective head also trains on the mirrored copies
# of its left/right images, labels swapped (the data lane's ``mirror.mirrored_rows``: human labels, text-free images,
# labels a flip changes), and answers every text-free image as the average of the image and its mirror, the mirror's
# left and right swapped back. The averaged answer is mirror-equivariant by construction: an image and its mirror get
# swapped answers. SigLIP's ``pooler_output`` is not mirror-invariant (#170), so the mirror carries real information.

#: ``HELPER_MIRROR=1`` turns the mirror-aware perspective head on for ``from_env`` (the live predictor).
MIRROR_ENV = "HELPER_MIRROR"


def mirrored_embeddings(
    pairs: Iterable[tuple[str, str]],
    encoder_name: str = DEFAULT_ENCODER,
    load_encoder: Callable[[str], Any] | None = None,
    directory: str | Path | None = None,
    batch_size: int = 16,
) -> dict[str, np.ndarray]:
    """``{image_id: embedding of the mirrored image}``: the same cached prepared JPEG the original embedding read,
    flipped pixel for pixel, embedded once and cached under ``cache/features/mirror/<encoder>/``."""
    from PIL import Image as PillowImage
    from PIL import ImageOps

    from galaxus.data import neighbours
    from galaxus.data.paths import cache_dir

    pairs = [(str(image_id), str(path)) for image_id, path in pairs]
    folder = Path(directory) if directory is not None else cache_dir("features", "mirror", encoder_name.replace("/", "__"))
    folder.mkdir(parents=True, exist_ok=True)
    with _lock_for(f"embed:{folder}"):
        missing = [(image_id, path) for image_id, path in pairs if not (folder / f"{image_id}.npy").is_file()]
        if missing:
            encoder = (load_encoder or shared_encoder)(encoder_name)
            for start in range(0, len(missing), batch_size):
                batch = missing[start : start + batch_size]
                images = [
                    ImageOps.mirror(PillowImage.open(io.BytesIO(neighbours.cached_jpeg(
                        image_id, path, image_size=neighbours.IMAGE_SIZE, patch_size=neighbours.PATCH_SIZE)))
                        .convert("RGB"))
                    for image_id, path in batch
                ]
                vectors = np.asarray(encoder.encode(images))
                for (image_id, _), vector in zip(batch, vectors, strict=True):
                    partial = folder / f"{image_id}.npy.part"
                    with partial.open("wb") as handle:  # through a handle: np.save would append ".npy" to a name
                        np.save(handle, np.asarray(vector, dtype=neighbours.EMBEDDING_DTYPE))
                    partial.replace(folder / f"{image_id}.npy")
        return {image_id: np.load(folder / f"{image_id}.npy").astype(np.float32).reshape(-1) for image_id, _ in pairs}


def fused_mirrored_embeddings(pairs: Iterable[tuple[str, str]], _name: str = FUSED_NAME) -> dict[str, np.ndarray]:
    """The fused features of the mirrored images, for ``helper-fused-mirror``'s perspective head."""
    return fused(mirrored_embeddings)(pairs, _name)


@dataclass(frozen=True)
class MirrorSetup:
    """Which dimension a mirror changes, how its classes swap, and which overlay answer means "no text"."""

    dimension: str
    swap: Callable[[str], str]
    text_dimension: str
    no_text: str
    #: The training rows worth mirroring (``mirror.mirrored_rows`` with the labels swapped).
    extra_rows: Callable[[pd.DataFrame], pd.DataFrame]

    @classmethod
    def from_classes(cls, allowed: Mapping[str, Iterable[str]], text_dimension: str | None = None,
                     no_text: str | None = None) -> MirrorSetup:
        from galaxus.data import classes, mirror

        dimension = mirror.mirrorable_dimension(allowed)
        text_dimension = text_dimension or classes.overlay_dimension()
        no_text = no_text or classes.no_overlay_class()
        return cls(
            dimension=dimension,
            swap=lambda label: mirror.mirror_class(dimension, label, allowed),
            text_dimension=text_dimension,
            no_text=no_text,
            extra_rows=lambda frame: mirror.mirrored_rows(frame, allowed=allowed, no_text_class=no_text,
                                                           dimension=dimension, text_dimension=text_dimension),
        )


def mirror_average(head: heads.Head, x: np.ndarray, m: np.ndarray, swap: Callable[[str], str]) -> np.ndarray:
    """Log-probabilities of each image averaged with its mirror's, the mirror's classes swapped back."""
    column = {name: k for k, name in enumerate(head.classes)}
    back = [column.get(swap(name), k) for k, name in enumerate(head.classes)]
    p = np.exp(head.log_probs(x)) + np.exp(head.log_probs(m))[:, back]
    return np.log(np.clip(0.5 * p, 1e-12, None))


def fit_mirrored_head(
    rows: pd.DataFrame,
    features: Mapping[str, np.ndarray],
    mirrored: Mapping[str, np.ndarray],
    setup: MirrorSetup,
    scored_mask: Callable[[pd.DataFrame, str], pd.Series],
    folds: Callable[[pd.DataFrame], Iterable[tuple[pd.DataFrame, pd.DataFrame]]] | None,
    seed: int = 0,
) -> tuple[heads.Head, dict[str, Any]]:
    """The perspective head on the original rows plus their mirrored copies, calibrated out of fold on answers that
    use the mirror exactly as prediction does (text-free images averaged with their mirror)."""
    d = setup.dimension

    def fit(part: pd.DataFrame) -> heads.Head:
        extra = setup.extra_rows(part)
        x = np.vstack([heads._matrix(features, part["image_id"]), heads._matrix(mirrored, extra["image_id"])])
        labels = part[d].astype(str).tolist() + extra[d].astype(str).tolist()
        return heads.fit_head(x, labels, seed=seed)

    scored = rows[scored_mask(rows, d)]
    head = fit(scored)
    report: dict[str, Any] = {"mirrored_training_rows": len(setup.extra_rows(scored))}
    if folds is None or len(head.classes) < 2:
        return head, report
    oof_ids: list[str] = []
    oof_rows: list[np.ndarray] = []
    for fit_part, held in folds(rows):
        fit_scored, held_scored = fit_part[scored_mask(fit_part, d)], held[scored_mask(held, d)]
        if fit_scored.empty or held_scored.empty:
            continue
        fold_head = fit(fit_scored)
        ids = held_scored["image_id"].astype(str)
        log_probs = fold_head.log_probs(heads._matrix(features, ids))
        text_free = (held_scored[setup.text_dimension].astype(str) == setup.no_text).to_numpy()
        if text_free.any():
            log_probs[text_free] = mirror_average(fold_head, heads._matrix(features, ids[text_free]),
                                                  heads._matrix(mirrored, ids[text_free]), setup.swap)
        full = np.full((len(held_scored), len(head.classes)), -1e9)
        for k, name in enumerate(fold_head.classes):
            full[:, head.classes.index(name)] = log_probs[:, k]
        oof_rows.append(full)
        oof_ids.extend(ids)
    if oof_rows:
        truth = scored.set_index(scored["image_id"].astype(str)).loc[oof_ids, d].astype(str)
        head.bias = fit_bias(np.vstack(oof_rows), np.searchsorted(head.classes, truth.to_numpy()))
        report["bias"] = dict(zip(head.classes, head.bias.round(2).tolist()))
    return head, report


class MirrorHelperPredictor(HelperPredictor):
    """The helper with the mirror-aware perspective head: text-free images answer perspective from the average of
    the image and its mirror (the helper's own overlay answer decides "text-free", as mirror TTA does, #48)."""

    name = "helper-mirror"

    def __init__(self, model: heads.HeadsModel, encoder_name: str, embed: Callable[..., dict[str, np.ndarray]],
                 mirror_embed: Callable[..., dict[str, np.ndarray]], setup: MirrorSetup) -> None:
        super().__init__(model, encoder_name, embed)
        self.mirror_embed = mirror_embed
        self.setup = setup
        self.mirror_features: dict[str, np.ndarray] = {}

    def predict(self, batch: Sequence[ImageItem]) -> list[Prediction]:
        predictions = super().predict(batch)
        setup, head = self.setup, self.heads_model.heads[self.setup.dimension]
        chosen = [k for k, p in enumerate(predictions) if p.labels.get(setup.text_dimension) == setup.no_text]
        if not chosen:
            return predictions
        items = [batch[k] for k in chosen]
        todo = [item for item in items if str(item.image_id) not in self.mirror_features]
        if todo:
            self.mirror_features.update(self.mirror_embed([(str(i.image_id), str(i.path)) for i in todo],
                                                          self.encoder_name))
        log_probs = mirror_average(head, np.stack([self._vector(item) for item in items]),
                                   np.stack([self.mirror_features[str(item.image_id)] for item in items]), setup.swap)
        index = apply_bias(log_probs, head.bias if head.bias.size else np.zeros(len(head.classes)))
        out = list(predictions)
        for row, k in enumerate(chosen):
            labels, confidence = dict(predictions[k].labels), dict(predictions[k].confidence or {})
            labels[setup.dimension] = head.classes[index[row]]
            confidence[setup.dimension] = float(np.exp(log_probs[row, index[row]]))
            out[k] = Prediction(image_id=predictions[k].image_id, labels=labels, confidence=confidence)
        return out

    def config(self) -> dict[str, object]:
        return {**super().config(), "mirror": {"dimension": self.setup.dimension, "averaged_when": "text-free answer",
                                               "training": "mirrored copies of the human left/right rows"}}


def factory(
    encoder_name: str = DEFAULT_ENCODER,
    embed: Callable[..., dict[str, np.ndarray]] = embeddings,
    mirror: bool = False,
    mirror_embed: Callable[..., dict[str, np.ndarray]] = mirrored_embeddings,
    setup: MirrorSetup | None = None,
):
    """The eval script's factory: heads fitted on ``rows`` (train minus the official holdout), then the predictor.

    ``mirror=True`` replaces the perspective head with the mirror-aware one (``helper-siglip-mirror``).
    """

    def build(rows: pd.DataFrame, data: Any, seed: int) -> HelperPredictor:
        pairs = list(zip(rows[ID_COLUMN].astype(str), rows[FILE_COLUMN].astype(str), strict=True))
        features = embed(pairs, encoder_name)
        model = heads.fit_heads(rows, features, data.dimensions(), data.scored_mask, folds=product_folds(seed),
                                key=f"helper:{encoder_name}", seed=seed)
        if not mirror:
            return HelperPredictor(model, encoder_name, embed)
        resolved = setup or MirrorSetup.from_classes(data.allowed_classes())
        scored = rows[data.scored_mask(rows, resolved.dimension)]
        text_free = scored[scored[resolved.text_dimension].astype(str) == resolved.no_text]
        wanted = pd.concat([text_free, resolved.extra_rows(scored)])[[ID_COLUMN, FILE_COLUMN]].drop_duplicates(ID_COLUMN)
        mirrored = mirror_embed(list(zip(wanted[ID_COLUMN].astype(str), wanted[FILE_COLUMN].astype(str), strict=True)),
                                encoder_name)
        head, report = fit_mirrored_head(rows, features, mirrored, resolved, data.scored_mask, product_folds(seed), seed)
        model.heads[resolved.dimension] = head
        model.meta.setdefault("calibration", {})[resolved.dimension] = report
        return MirrorHelperPredictor(model, encoder_name, embed, mirror_embed, resolved)

    return build


def _flag(environ: Mapping[str, str], name: str) -> bool:
    return environ.get(name, "") not in ("", "0")


def _training_rows(data: Any, environ: Mapping[str, str]) -> tuple[pd.DataFrame, str]:
    """Train minus the official holdout, or with ``HELPER_FIT_ALL=1`` every labelled row (dev and final only)."""
    train = data.read_train()
    if _flag(environ, FIT_ALL_ENV):
        return train, "every labelled row, the official holdout included (HELPER_FIT_ALL)"
    return data.split_holdout(train)[0], "train minus the official holdout"


def _live(build: Callable[..., HelperPredictor], environ: Mapping[str, str] | None) -> HelperPredictor:
    from galaxus.model.eval import DataLane

    environ = os.environ if environ is None else environ
    data = DataLane.real()
    rows, fitted_on = _training_rows(data, environ)
    predictor = build(rows, data, 0)
    predictor.heads_model.meta["fitted_on"] = fitted_on
    return predictor


def from_env(environ: Mapping[str, str] | None = None) -> HelperPredictor:
    """For the runner, ``submit --run`` and ``final_run --predictor``: SigLIP base heads, fitted on train minus the
    official holdout. ``HELPER_MIRROR=1`` uses the mirror-aware perspective head; ``HELPER_FIT_ALL=1`` fits on every
    labelled row."""
    environ = os.environ if environ is None else environ
    return _live(factory(mirror=_flag(environ, MIRROR_ENV)), environ)


#: ``HELPER_FUSED_MIRROR=1`` turns the mirror-aware perspective head on for :func:`fused_from_env` only, so the
#: SigLIP member and the fused member of one live ensemble can differ.
FUSED_MIRROR_ENV = "HELPER_FUSED_MIRROR"


def fused_factory(mirror: bool = False):
    """``helper-fused`` (EXP-0059): #51's heads on the three encoders' features at unit length, concatenated."""
    return factory(encoder_name=FUSED_NAME, embed=fused_embeddings, mirror=mirror,
                   mirror_embed=fused_mirrored_embeddings)


def fused_from_env(environ: Mapping[str, str] | None = None) -> HelperPredictor:
    """``helper-fused`` for the runner and the live ensemble (``fused=galaxus.model.helper:fused_from_env``).
    ``HELPER_FUSED_MIRROR=1`` uses the mirror-aware perspective head; ``HELPER_FIT_ALL=1`` fits on every labelled
    row."""
    environ = os.environ if environ is None else environ
    return _live(fused_factory(mirror=_flag(environ, FUSED_MIRROR_ENV)), environ)
