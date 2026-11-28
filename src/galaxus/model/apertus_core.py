"""Apertus as the core, the image encoders as the base (#214): one head per dimension on Apertus v1.5's hidden state
of each image read together with its catalog text, with the frozen encoders' vectors appended.

* **The core:** ``galaxus.model.heads``' feature store, filled on a GPU by ``python -m galaxus.model.heads extract``
  (``train/hf_job_apertus_features_start.sh``): Apertus v1.5 8B, unchanged, one forward pass per image with
  ``heads.prompt_for``'s prompt (the catalog fields), the last token's hidden state at ``APERTUS_CORE_LAYERS``
  (default ``-1,-9``). Each layer's vector is put at unit length.
* **The base:** ``helper.fused_embeddings``, EXP-0059's frozen SigLIP base, DINOv2 base and SigLIP 2 base, each at unit
  length, computed where the predictor runs, as ``helper-fused`` does.
* **An optional block:** ``APERTUS_CORE_BASE_SCORES=<npz>``, written with :func:`save_base_scores`: one row of another
  model's class scores per image, for example the fine-tuned SigLIP 2 of #202. **The rows the heads train on must be
  out of fold** (from a model that never saw that row's product): in-sample scores look nearly perfect on train, so the
  heads would lean on them and ignore Apertus, and the holdout would drop.
* **The heads:** #51's ``fit_heads`` through ``helper.factory``: tempered class weights, product-fold calibration.

Heads on Apertus hidden states count as "Apertus classifying" (Q-0007, decisions.md); the frozen encoders are the
helpers of Q-0008. The optional block waits for the hosts' answer on a fine-tuned encoder as an input (#202, #214).

    uv run --extra local python -m galaxus.model.eval apertus-core --exp-id EXP-0093
    APERTUS_CORE_BASE_SCORES=$GALAXUS_DATA/out/apertus_core/ft.npz \\
        uv run --extra local python -m galaxus.model.eval apertus-core --exp-id EXP-0094

For the live ensemble and the final run: ``core=galaxus.model.apertus_core:from_env`` (``HELPER_FIT_ALL=1`` fits on every
labelled row). The final images' Apertus features must be in the store first (#214 step 6).
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from galaxus.model import heads, helper

NAME = "apertus-core"
APERTUS_8B = "swiss-ai/Apertus-v1.5-8B"
LAYERS_ENV = "APERTUS_CORE_LAYERS"
POOL_ENV = "APERTUS_CORE_POOL"
BASE_SCORES_ENV = "APERTUS_CORE_BASE_SCORES"
#: The last layer and about three quarters of the depth (layer 24 of 32), where another team found frozen Apertus 8B
#: heads best (#145). The feature job writes both by default.
DEFAULT_LAYERS = (-1, -9)
DEFAULT_POOL = "last"


def parse_layers(text: str) -> tuple[int, ...]:
    """``"-1,-9"`` -> ``(-1, -9)``; empty -> :data:`DEFAULT_LAYERS`."""
    return tuple(int(part) for part in text.split(",") if part.strip()) or DEFAULT_LAYERS


def per_layer_unit(vector: np.ndarray, layers: int) -> np.ndarray:
    """An Apertus feature vector (``layers`` x hidden size, flattened) with each layer's part at unit length."""
    blocks = np.asarray(vector, dtype=np.float32).reshape(layers, -1)
    return np.concatenate([helper.unit_length(block) for block in blocks])


class ApertusFeatures:
    """The Apertus hidden states in ``heads``' feature store, loaded once and reloaded when an image is missing (the
    final part's features arrive after the predictor is built)."""

    def __init__(self, store: heads.FeatureStore, layers: Sequence[int] = DEFAULT_LAYERS, model: str = APERTUS_8B,
                 pool: str = DEFAULT_POOL) -> None:
        self.store, self.layer_ids, self.model, self.pool = store, tuple(int(x) for x in layers), model, pool
        self._vectors: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def get(self, ids: Sequence[str]) -> dict[str, np.ndarray]:
        with self._lock:
            if any(str(i) not in self._vectors for i in ids):
                self._vectors = self.store.load()
            missing = [str(i) for i in ids if str(i) not in self._vectors]
            if missing:
                raise KeyError(
                    f"{len(missing)} image(s) have no Apertus features in {self.store.root}: extract their part on a "
                    "GPU first (train/hf_job_apertus_features_start.sh, or `python -m galaxus.model.heads extract "
                    "--part <part> --layers=<layers> --pool <pool>`) and untar the features under the cache")
            return {str(i): per_layer_unit(self._vectors[str(i)], len(self.layer_ids)) for i in ids}


def features_from_env(environ: Mapping[str, str] | None = None) -> ApertusFeatures:
    """The store the feature job filled: ``$APERTUS_MODEL``, ``$APERTUS_CORE_LAYERS``, ``$APERTUS_CORE_POOL``."""
    environ = os.environ if environ is None else environ
    model = environ.get(heads.APERTUS_ENV, "") or APERTUS_8B
    layers = parse_layers(environ.get(LAYERS_ENV, ""))
    pool = environ.get(POOL_ENV, "") or DEFAULT_POOL
    return ApertusFeatures(heads.store_for(model, layers, pool), layers, model, pool)


# ---------------------------------------------------------------- the optional block: another model's scores


def save_base_scores(path: str | os.PathLike[str], ids: Iterable[object], scores: Any) -> Path:
    """Write one row of scores per image (any width, e.g. a model's log-probabilities over every class, clipped) for
    ``$APERTUS_CORE_BASE_SCORES``. Rows of training images must be out of fold (module docstring). Outside git: under
    ``$GALAXUS_DATA/out/``."""
    names = np.asarray([str(i) for i in ids])
    matrix = np.asarray(scores, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) != len(names):
        raise ValueError(f"scores must have one row per image: {matrix.shape} for {len(names)} ids")
    if len(set(names.tolist())) != len(names):
        raise ValueError("an image id appears twice")
    if not np.isfinite(matrix).all():
        raise ValueError("the scores hold NaN or inf: clip log-probabilities first (e.g. at -20)")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with partial.open("wb") as handle:  # through a handle, so numpy adds no ".npz" to the temporary name
        np.savez(handle, ids=names, scores=matrix)
    partial.replace(target)
    return target


class BaseScores:
    """The rows of :func:`save_base_scores`' file, reloaded when an image is missing (final rows are added at 13:00)."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._rows: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def _load(self) -> dict[str, np.ndarray]:
        with np.load(self.path, allow_pickle=False) as data:
            return {str(i): row for i, row in zip(data["ids"], data["scores"].astype(np.float32), strict=True)}

    def get(self, ids: Sequence[str]) -> dict[str, np.ndarray]:
        with self._lock:
            if any(str(i) not in self._rows for i in ids):
                self._rows = self._load()
            missing = [str(i) for i in ids if str(i) not in self._rows]
            if missing:
                raise KeyError(f"{len(missing)} image(s) have no row in {self.path}: add them with save_base_scores")
            return {str(i): self._rows[str(i)] for i in ids}


def base_scores_from_env(environ: Mapping[str, str] | None = None) -> BaseScores | None:
    environ = os.environ if environ is None else environ
    path = environ.get(BASE_SCORES_ENV, "")
    return BaseScores(path) if path else None


# ---------------------------------------------------------------- features, heads, predictor


def core_embeddings(
    apertus: ApertusFeatures,
    base: Callable[..., dict[str, np.ndarray]] | None = helper.fused_embeddings,
    scores: BaseScores | None = None,
) -> Callable[..., dict[str, np.ndarray]]:
    """The heads' features per image: Apertus' layers at unit length, then the base encoders, then the optional
    scores, always in that order."""

    def embed(pairs: Iterable[tuple[str, str]], _name: str = NAME) -> dict[str, np.ndarray]:
        pairs = [(str(image_id), str(path)) for image_id, path in pairs]
        ids = [image_id for image_id, _ in pairs]
        blocks = [apertus.get(ids)]
        if base is not None:
            blocks.append(base(pairs))
        if scores is not None:
            blocks.append(scores.get(ids))
        return {image_id: np.concatenate([block[image_id] for block in blocks]) for image_id in ids}

    return embed


def factory(
    apertus: ApertusFeatures | None = None,
    base: Callable[..., dict[str, np.ndarray]] | None = helper.fused_embeddings,
    base_name: str | None = helper.FUSED_NAME,
    scores: BaseScores | None = None,
    environ: Mapping[str, str] | None = None,
):
    """The eval script's factory: heads on :func:`core_embeddings` of ``rows`` (train minus the official holdout)."""

    def build(rows: Any, data: Any, seed: int) -> helper.HelperPredictor:
        env = os.environ if environ is None else environ
        features = apertus or features_from_env(env)
        extra = scores if scores is not None else base_scores_from_env(env)
        predictor = helper.factory(encoder_name=NAME, embed=core_embeddings(features, base, extra))(rows, data, seed)
        # What the EXP row records: Apertus classifies; the encoders (and the optional scores) are its helpers.
        encoders = helper.COMPONENTS.get(base_name, (base_name,)) if base is not None and base_name else ()
        predictor.model = features.model
        predictor.model_revision = None  # Apertus unchanged, as the gateway serves it: the commit names the method
        predictor.helpers = tuple(encoders) + ((f"base-scores:{extra.path.name}",) if extra is not None else ())
        predictor.heads_model.meta.update({
            "core": features.model, "layers": list(features.layer_ids), "pool": features.pool,
            "base": base_name if base is not None else None,
            "base_revisions": [helper.REVISIONS.get(name) for name in encoders],
            "base_scores": str(extra.path) if extra is not None else None,
        })
        return predictor

    return build


def from_env(environ: Mapping[str, str] | None = None) -> helper.HelperPredictor:
    """For the runner, ``submit --run`` and the live ensemble (``core=galaxus.model.apertus_core:from_env``): fitted on
    train minus the official holdout, or on every labelled row with ``HELPER_FIT_ALL=1``."""
    environ = os.environ if environ is None else environ
    return helper._live(factory(environ=environ), environ)
