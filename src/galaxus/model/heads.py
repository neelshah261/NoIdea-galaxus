"""#41: Apertus reads each image once at 512 px; small heads per dimension give the six answers (plan #40, option 1).

Three steps, each resumable and each its own command:

1. **extract**: Apertus v1.5 (unchanged) gets the 512 px image from the data lane's cache (#27) and one fixed prompt
   with the product context; we keep the hidden state of the last prompt token (``LocalBackend.forward``). Stored as
   shards under ``$GALAXUS_DATA/cache/features/<key>/<part>/``. One forward pass per image, no text generated.
2. **fit**: per dimension, a softmax head (numpy, CPU) on the scored human labels of train minus the holdout, with
   tempered class weights for macro F1. Out-of-fold probabilities over the grouped folds feed the per-class
   calibration (#42); the final heads are refit on all training rows. Scored on the official holdout through
   ``model.eval`` and written as an EXP row.
3. **predict**: :class:`HeadsPredictor` implements the contract, so the serve runner, submit and the Saturday
   final run use it unchanged (``--predictor galaxus.model.heads:predictor_from_env``).

**Open with the hosts:** whether heads on Apertus hidden states count as "Apertus performing the classification"
(Q-0007, #40). Ship this as the main method only after their yes; until then it is an experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from galaxus.model.calibration import apply_bias, fit_bias
from galaxus.types import ImageItem, Prediction

#: Bump when the prompt text changes: features from another prompt are another feature set.
PROMPT_VERSION = "v1"
PROMPT = (
    "Product image from an online shop.{context} Classify the image: what it shows, its background, the camera "
    "perspective, whether a real person or animal is present, any text overlay and the language of that text."
)
#: Product-context fields, in the order they are written into the prompt (Kaggle data page, "Columns").
CONTEXT_FIELDS = ("product_title", "product_type", "brand", "category", "subcategory")
IMAGE_SIZE = 512  # Swarup's condition: always 512 px of area
SHARD_SIZE = 256
#: Per part, the ids ``extract`` could not read, one per line (never printed). ``apertus_core`` treats exactly these as
#: unreadable images (#219), so a forgotten extraction still stops a run instead of falling back.
UNREADABLE_FILE = "unreadable.txt"
DEFAULT_LAYERS = (-1,)
DEFAULT_POOL = "last"
EPOCHS = 300
LEARNING_RATE = 0.01
L2 = 1e-3
CLASS_WEIGHT_POWER = 0.5  # 0 = no reweighting, 1 = fully balanced; 0.5 tempers it for macro F1 without wrecking
MODEL_ENV = "HEADS_MODEL"
APERTUS_ENV = "APERTUS_MODEL"


# ---------------------------------------------------------------- prompt and features


def prompt_for(context: Mapping[str, str] | None) -> str:
    """The fixed prompt with this image's product context, or none when the context is missing."""
    if not context:
        return PROMPT.format(context="")
    parts = [f"{name.replace('_', ' ')}: {context[name]}" for name in CONTEXT_FIELDS if context.get(name)]
    extra = [f"{name}: {value}" for name, value in context.items() if name not in CONTEXT_FIELDS and value]
    text = "; ".join(parts + extra)
    return PROMPT.format(context=f" {text}." if text else "")


def feature_key(model: str, layers: Sequence[int], pool: str, image_size: int = IMAGE_SIZE) -> str:
    """A folder name that changes whenever anything that changes the features changes."""
    slug = model.rsplit("/", 1)[-1]
    layer_text = "_".join(str(layer) for layer in layers)
    return f"{slug}-{image_size}px-{PROMPT_VERSION}-L{layer_text}-{pool}"


@dataclass
class FeatureStore:
    """Hidden states by image id, in npz shards under ``<root>/<part>/``. Appending is crash-safe."""

    root: Path

    def part_dir(self, part: str) -> Path:
        path = self.root / part
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ids(self, part: str) -> set[str]:
        return set(self.load(part))

    def unreadable(self, part: str | None = None) -> set[str]:
        """The ids ``extract`` skipped as unreadable, in one part or in every part."""
        files = [self.root / part / UNREADABLE_FILE] if part else sorted(self.root.glob(f"*/{UNREADABLE_FILE}"))
        return {line.strip() for file in files if file.is_file() for line in file.read_text().splitlines() if line.strip()}

    def mark_unreadable(self, part: str, ids: Sequence[str]) -> Path:
        """Add ``ids`` to the part's unreadable list, through a temporary name (crash-safe like the shards)."""
        target = self.part_dir(part) / UNREADABLE_FILE
        merged = sorted(self.unreadable(part) | {str(i) for i in ids})
        partial = target.with_name(target.name + ".part")
        partial.write_text("".join(f"{image_id}\n" for image_id in merged))
        os.replace(partial, target)
        return target

    def load(self, part: str | None = None) -> dict[str, np.ndarray]:
        """``{image_id: features}`` of one part, or of every part when ``part`` is None."""
        parts = [part] if part else [p.name for p in self.root.iterdir() if p.is_dir()] if self.root.exists() else []
        features: dict[str, np.ndarray] = {}
        for name in parts:
            for shard in sorted((self.root / name).glob("shard_*.npz")):
                with np.load(shard, allow_pickle=False) as data:
                    for image_id, vector in zip(data["ids"], data["x"]):
                        features[str(image_id)] = vector
        return features

    def append(self, part: str, ids: Sequence[str], x: np.ndarray) -> Path:
        """Write one shard (float16) through a temporary name, so a crash never leaves half a shard."""
        folder = self.part_dir(part)
        number = len(list(folder.glob("shard_*.npz")))
        target = folder / f"shard_{number:05d}.npz"
        temporary = folder / f".shard_{number:05d}.tmp.npz"
        np.savez(temporary, ids=np.asarray(ids, dtype=str), x=x.astype(np.float16))
        os.replace(temporary, target)
        return target


def extract(
    forward: Callable[[Sequence[Any], str], np.ndarray],
    items: Sequence[ImageItem],
    store: FeatureStore,
    part: str,
    image_bytes: Callable[[ImageItem], bytes],
    shard_size: int = SHARD_SIZE,
    log: Callable[[str], None] | None = print,
) -> int:
    """Hidden states for every item not yet in ``store``; returns how many were computed.

    ``forward(images, prompt) -> (images, layers, hidden)`` is ``LocalBackend.forward`` with the layers and pool
    bound; ``image_bytes(item)`` gives the 512 px prepared JPEG (data lane cache).

    An image that cannot be read or prepared (``OSError``, which covers PIL's unidentified and truncated files, or
    ``ValueError``) is skipped and counted, never named, so one bad file in a new batch cannot stop the run (#218); the
    reference notebook keeps the constant answer for such an image. Errors of ``forward`` itself still stop it.
    """
    done = store.ids(part)
    todo = [item for item in items if str(item.image_id) not in done]
    if log and done:
        log(f"[heads] {part}: {len(done)} already extracted, {len(todo)} to go")
    started = time.perf_counter()
    ids: list[str] = []
    rows: list[np.ndarray] = []
    computed = 0
    unreadable: list[str] = []
    for item in todo:
        try:
            image = image_bytes(item)
        except (OSError, ValueError):
            unreadable.append(str(item.image_id))
            continue
        vector = forward([image], prompt_for(item.context))[0]
        ids.append(str(item.image_id))
        rows.append(np.asarray(vector, dtype=np.float32).reshape(-1))
        if len(ids) == shard_size:
            store.append(part, ids, np.stack(rows))
            computed += len(ids)
            ids, rows = [], []
            if log:
                speed = computed / (time.perf_counter() - started)
                log(f"[heads] {part}: {len(done) + computed}/{len(items)} ({speed:.2f} img/s)")
    if ids:
        store.append(part, ids, np.stack(rows))
        computed += len(ids)
    if unreadable:
        store.mark_unreadable(part, unreadable)  # in the store, so it travels with the features; never in a log
        if log:
            log(f"[heads] {part}: {len(unreadable)} image(s) could not be read and were skipped; their rows keep the "
                "constant answer")
    return computed


# ---------------------------------------------------------------- heads


@dataclass
class Head:
    """Softmax regression for one dimension, on standardised features."""

    classes: list[str]
    weights: np.ndarray
    intercept: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    bias: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def log_probs(self, x: np.ndarray) -> np.ndarray:
        logits = ((x - self.mean) / self.scale) @ self.weights + self.intercept
        logits -= logits.max(axis=1, keepdims=True)
        return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))

    def predict(self, x: np.ndarray) -> tuple[list[str], np.ndarray]:
        """(class per row, probability of that class before the bias)."""
        log_probs = self.log_probs(x)
        bias = self.bias if self.bias.size else np.zeros(len(self.classes))
        index = apply_bias(log_probs, bias)
        return [self.classes[i] for i in index], np.exp(log_probs[np.arange(len(index)), index])


def fit_head(
    x: np.ndarray,
    labels: Sequence[str],
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    l2: float = L2,
    weight_power: float = CLASS_WEIGHT_POWER,
    seed: int = 0,
) -> Head:
    """Full-batch Adam on weighted cross-entropy. Deterministic for a given seed."""
    classes = sorted(set(labels))
    target = np.searchsorted(classes, np.asarray(labels))
    mean = x.mean(axis=0)
    scale = x.std(axis=0) + 1e-6
    z = ((x - mean) / scale).astype(np.float32)
    n, d = z.shape
    k = len(classes)
    if k == 1:
        return Head(classes, np.zeros((d, 1), np.float32), np.zeros(1, np.float32), mean, scale)
    counts = np.bincount(target, minlength=k).astype(np.float64)
    class_weight = (n / (k * counts)) ** weight_power
    sample_weight = (class_weight[target] / class_weight[target].mean()).astype(np.float32)
    rng = np.random.default_rng(seed)
    weights = (rng.standard_normal((d, k)) * 0.001).astype(np.float32)
    intercept = np.zeros(k, np.float32)
    onehot = np.eye(k, dtype=np.float32)[target]
    moments = [np.zeros_like(weights), np.zeros_like(weights), np.zeros_like(intercept), np.zeros_like(intercept)]
    beta1, beta2 = 0.9, 0.999
    for step in range(1, epochs + 1):
        logits = z @ weights + intercept
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        delta = (probs - onehot) * sample_weight[:, None] / n
        grad_w = z.T @ delta + l2 * weights
        grad_b = delta.sum(axis=0)
        for value, grad, first, second in ((weights, grad_w, 0, 1), (intercept, grad_b, 2, 3)):
            moments[first] = beta1 * moments[first] + (1 - beta1) * grad
            moments[second] = beta2 * moments[second] + (1 - beta2) * grad * grad
            corrected = moments[first] / (1 - beta1**step)
            value -= learning_rate * corrected / (np.sqrt(moments[second] / (1 - beta2**step)) + 1e-8)
    return Head(classes, weights, intercept, mean, scale)


@dataclass
class HeadsModel:
    """One head per dimension plus what produced its features."""

    heads: dict[str, Head]
    key: str
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> Path:
        arrays: dict[str, np.ndarray] = {}
        for dimension, head in self.heads.items():
            for name in ("weights", "intercept", "mean", "scale", "bias"):
                arrays[f"{dimension}.{name}"] = getattr(head, name)
            arrays[f"{dimension}.classes"] = np.asarray(head.classes, dtype=str)
        arrays["__meta__"] = np.asarray(json.dumps({"key": self.key, **self.meta}))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> HeadsModel:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["__meta__"]))
            dimensions = sorted({name.split(".")[0] for name in data.files if name != "__meta__"})
            heads = {
                dimension: Head(
                    classes=[str(c) for c in data[f"{dimension}.classes"]],
                    weights=data[f"{dimension}.weights"],
                    intercept=data[f"{dimension}.intercept"],
                    mean=data[f"{dimension}.mean"],
                    scale=data[f"{dimension}.scale"],
                    bias=data[f"{dimension}.bias"],
                )
                for dimension in dimensions
            }
        return cls(heads=heads, key=meta.pop("key"), meta=meta)


def _matrix(features: Mapping[str, np.ndarray], ids: Iterable[str]) -> np.ndarray:
    return np.stack([np.asarray(features[str(i)], dtype=np.float32).reshape(-1) for i in ids])


def fit_heads(
    rows: pd.DataFrame,
    features: Mapping[str, np.ndarray],
    dimensions: Sequence[str],
    scored_mask: Callable[[pd.DataFrame, str], pd.Series],
    folds: Callable[[pd.DataFrame], Iterable[tuple[pd.DataFrame, pd.DataFrame]]] | None = None,
    key: str = "",
    seed: int = 0,
    log: Callable[[str], None] | None = print,
    calibration_mask: Callable[[pd.DataFrame, str], pd.Series] | None = None,
) -> HeadsModel:
    """Heads on the scored human labels of ``rows``; with ``folds``, calibrated on out-of-fold probabilities.

    ``scored_mask`` picks the training cells. ``calibration_mask`` (default: the same) picks the held-out cells the
    calibration bias is fitted on: D (#211) trains on human and teacher cells but calibrates on human cells only.
    Rows without features are an error: extract them first, so no image is silently left out of training.
    """
    calibration_mask = calibration_mask or scored_mask
    missing = [str(i) for i in rows["image_id"] if str(i) not in features]
    if missing:
        raise KeyError(f"{len(missing)} training image(s) have no features, first: {missing[:3]}; run extract first")
    heads: dict[str, Head] = {}
    report: dict[str, Any] = {}
    for dimension in dimensions:
        scored = rows[scored_mask(rows, dimension)]
        if scored.empty:
            raise ValueError(f"no scored human label for {dimension}")
        head = fit_head(_matrix(features, scored["image_id"]), scored[dimension].astype(str).tolist(), seed=seed)
        if folds is not None and len(head.classes) > 1:
            oof_ids: list[str] = []
            oof_rows: list[np.ndarray] = []
            for fit_part, held in folds(rows):
                fit_scored = fit_part[scored_mask(fit_part, dimension)]
                held_scored = held[calibration_mask(held, dimension)]
                if held_scored.empty or fit_scored.empty:
                    continue
                fold_head = fit_head(
                    _matrix(features, fit_scored["image_id"]), fit_scored[dimension].astype(str).tolist(), seed=seed
                )
                fold_lp = fold_head.log_probs(_matrix(features, held_scored["image_id"]))
                full = np.full((len(held_scored), len(head.classes)), -1e9)
                for column, name in enumerate(fold_head.classes):
                    full[:, head.classes.index(name)] = fold_lp[:, column]
                oof_rows.append(full)
                oof_ids.extend(str(i) for i in held_scored["image_id"])
            if oof_rows:
                calibrated = rows[calibration_mask(rows, dimension)]
                truth = calibrated.set_index(calibrated["image_id"].astype(str)).loc[oof_ids, dimension].astype(str)
                true_index = np.searchsorted(head.classes, truth.to_numpy())
                head.bias = fit_bias(np.vstack(oof_rows), true_index)
                report[dimension] = {"bias": dict(zip(head.classes, head.bias.round(2).tolist()))}
        heads[dimension] = head
        if log:
            log(f"[heads] {dimension}: {len(scored)} scored rows, {len(head.classes)} classes")
    return HeadsModel(heads=heads, key=key, meta={"calibration": report, "seed": seed})


def out_of_fold_records(
    rows: pd.DataFrame,
    features: Mapping[str, np.ndarray],
    dimensions: Sequence[str],
    scored_mask: Callable[[pd.DataFrame, str], pd.Series],
    folds: Callable[[pd.DataFrame], Iterable[tuple[pd.DataFrame, pd.DataFrame]]],
    seed: int = 0,
) -> list[dict[str, Any]]:
    """One ``Prediction`` record per row of ``rows``, each predicted by heads that never saw that row's product.

    This is the input of the data lane's label-error finder (#7, ``galaxus.data.label_errors``): a confident
    out-of-fold answer that disagrees with a human label is a suspected label error, for the bonus. Heads are trained
    on scored human labels only; no calibration, so the confidence is the head's own probability.
    """
    records: dict[str, dict[str, Any]] = {
        str(i): {"image_id": str(i), "labels": {}, "confidence": {}} for i in rows["image_id"]
    }
    for fit_part, held in folds(rows):
        held_ids = [str(i) for i in held["image_id"]]
        x_held = _matrix(features, held_ids)
        for dimension in dimensions:
            fit_scored = fit_part[scored_mask(fit_part, dimension)]
            if fit_scored.empty:
                continue
            head = fit_head(
                _matrix(features, fit_scored["image_id"]), fit_scored[dimension].astype(str).tolist(), seed=seed
            )
            answers, confidence = head.predict(x_held)
            for image_id, answer, probability in zip(held_ids, answers, confidence):
                records[image_id]["labels"][dimension] = answer
                records[image_id]["confidence"][dimension] = float(probability)
    return [record for record in records.values() if record["labels"]]


def write_records(records: Iterable[Mapping[str, Any]], path: Path) -> Path:
    """Prediction records as jsonl (outside git: under ``$GALAXUS_DATA/out/``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------- predictor


class HeadsPredictor:
    """The contract's ``Predictor``: features from the store, or one Apertus forward pass when missing."""

    name = "heads"

    def __init__(
        self,
        model: HeadsModel,
        features: Mapping[str, np.ndarray] | None = None,
        compute: Callable[[ImageItem], np.ndarray] | None = None,
    ) -> None:
        self.heads_model = model
        #: What the EXP row records as ``model`` (eval reads ``predictor.model``): the Apertus checkpoint.
        self.model = str(model.meta.get("apertus", "")) or None
        self.model_revision = model.meta.get("revision")
        self.features = dict(features or {})
        self.compute = compute

    def _vector(self, item: ImageItem) -> np.ndarray:
        key = str(item.image_id)
        if key not in self.features:
            if self.compute is None:
                raise KeyError(f"no features for {key} and no Apertus backend to compute them")
            self.features[key] = np.asarray(self.compute(item), dtype=np.float32)
        return np.asarray(self.features[key], dtype=np.float32).reshape(-1)

    def predict(self, batch: Sequence[ImageItem]) -> list[Prediction]:
        x = np.stack([self._vector(item) for item in batch]).astype(np.float32)
        answers: dict[str, list[str]] = {}
        confidence: dict[str, np.ndarray] = {}
        for dimension, head in self.heads_model.heads.items():
            answers[dimension], confidence[dimension] = head.predict(x)
        return [
            Prediction(
                image_id=item.image_id,
                labels={d: answers[d][row] for d in answers},
                confidence={d: float(confidence[d][row]) for d in answers},
            )
            for row, item in enumerate(batch)
        ]

    def config(self) -> dict[str, object]:
        return {"key": self.heads_model.key, **self.heads_model.meta}


# ---------------------------------------------------------------- wiring to the real data and Apertus


def default_layers() -> tuple[int, ...]:
    raw = os.environ.get("HEADS_LAYERS", "")
    return tuple(int(x) for x in raw.split(",") if x.strip()) or DEFAULT_LAYERS


def store_for(model_name: str, layers: Sequence[int], pool: str) -> FeatureStore:
    from galaxus.data.paths import cache_dir

    return FeatureStore(cache_dir("features", feature_key(model_name, layers, pool)))


def _backend_forward(model_name: str, layers: Sequence[int], pool: str) -> Callable[[Sequence[Any], str], np.ndarray]:
    from galaxus.model.apertus.local import LocalBackend

    backend = LocalBackend(model_name)
    return lambda images, prompt: backend.forward(images, prompt, layers=tuple(layers), pool=pool)


def _cached_image(item: ImageItem) -> bytes:
    from galaxus.data.images import cached_jpeg

    return cached_jpeg(item.image_id, item.path, image_size=IMAGE_SIZE)


def predictor_from_env() -> HeadsPredictor:
    """For the serve runner: ``$HEADS_MODEL`` is the saved heads; missing features are computed with Apertus."""
    path = os.environ.get(MODEL_ENV)
    if not path:
        raise RuntimeError(f"set ${MODEL_ENV} to a heads file written by `python -m galaxus.model.heads fit`")
    model = HeadsModel.load(path)
    meta_layers = tuple(model.meta.get("layers", DEFAULT_LAYERS))
    pool = model.meta.get("pool", DEFAULT_POOL)
    model_name = model.meta.get("apertus", os.environ.get(APERTUS_ENV, "swiss-ai/Apertus-v1.5-8B"))
    store = store_for(model_name, meta_layers, pool)
    forward: list[Callable[[Sequence[Any], str], np.ndarray]] = []

    def compute(item: ImageItem) -> np.ndarray:
        if not forward:
            forward.append(_backend_forward(model_name, meta_layers, pool))
        return forward[0]([_cached_image(item)], prompt_for(item.context))[0]

    return HeadsPredictor(model, store.load(), compute)


def extract_source(part: str | None, csv: str | None) -> tuple[pd.DataFrame, str]:
    """The rows to extract and the store part they go to: a competition part, or ``csv-<file stem>`` for a new batch."""
    from galaxus.data import load

    if csv:
        return load.read_part(csv), f"csv-{Path(csv).stem}"
    reader = {"train": load.read_train, "dev": load.read_dev, "final": load.read_final}[part or ""]
    return reader(), str(part)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m galaxus.model.heads extract|fit|label-errors``."""
    parser = argparse.ArgumentParser(prog="python -m galaxus.model.heads", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    extract_cmd = sub.add_parser("extract", help="Apertus hidden states for one part, or any table of images (GPU)")
    source = extract_cmd.add_mutually_exclusive_group(required=True)
    source.add_argument("--part", choices=("train", "dev", "final"))
    source.add_argument("--csv", help="any new batch: a table with image_id and image_path (relative to it) and, "
                        "optionally, the catalog columns, read as dev.csv is (#218)")
    extract_cmd.add_argument("--limit", type=int, default=None)
    fit_cmd = sub.add_parser("fit", help="fit + calibrate heads on CPU, score the holdout, write the EXP row")
    fit_cmd.add_argument("--exp-id", default=None)
    fit_cmd.add_argument("--no-calibration", action="store_true")
    fit_cmd.add_argument("--allow-dirty", action="store_true")
    fit_cmd.add_argument("--note", default="")
    errors_cmd = sub.add_parser("label-errors", help="out-of-fold heads over all of train -> the bonus csv (#7)")
    errors_cmd.add_argument("--threshold", type=float, default=None, help="minimum confidence (data lane default)")
    errors_cmd.add_argument("--scored-only", action="store_true", help="only cells the metric scores")
    for command in (extract_cmd, fit_cmd, errors_cmd):
        command.add_argument("--model", default=os.environ.get(APERTUS_ENV, "swiss-ai/Apertus-v1.5-8B"))
        command.add_argument("--layers", default=",".join(map(str, default_layers())), help="e.g. -1 or -1,-9")
        command.add_argument("--pool", default=DEFAULT_POOL, choices=("last", "mean"))
    arguments = parser.parse_args(argv)
    layers = tuple(int(x) for x in arguments.layers.split(","))
    store = store_for(arguments.model, layers, arguments.pool)

    from galaxus.data import load
    from galaxus.serve.items import items_from_frame

    if arguments.command == "extract":
        frame, part = extract_source(arguments.part, arguments.csv)
        items = items_from_frame(frame, arguments.limit)
        forward = _backend_forward(arguments.model, layers, arguments.pool)
        computed = extract(forward, items, store, part, _cached_image)
        print(f"[heads] {part}: computed {computed}; store {store.root}")
        return 0

    from galaxus.data import split
    from galaxus.data.paths import out_dir
    from galaxus.model import eval as model_eval

    features = store.load()
    key = feature_key(arguments.model, layers, arguments.pool)

    if arguments.command == "label-errors":
        from galaxus.data import classes, label_errors

        train = load.read_train()
        records = out_of_fold_records(train, features, classes.dimensions(), split.scored_mask, split.iter_product_folds)
        oof_path = write_records(records, out_dir() / "predictions" / f"heads-oof-{key}" / "train_oof.jsonl")
        options: dict[str, Any] = {"scored_only": arguments.scored_only}
        if arguments.threshold is not None:
            options["threshold"] = arguments.threshold
        report = label_errors.find_label_errors(oof_path, train, **options)
        target = label_errors.write_label_errors(report)
        print(f"[heads] {len(records)} out-of-fold predictions -> {oof_path}")
        print(f"[heads] {len(report)} suspected label errors -> {target}")
        print("Never commit it. A human sends it to the Galaxus hosts by Discord DM (challenge_description.md).")
        return 0
    saved: dict[str, Path] = {}

    def factory(training_rows: pd.DataFrame, data: Any, seed: int) -> HeadsPredictor:
        heads = fit_heads(
            training_rows,
            features,
            data.dimensions(),
            data.scored_mask,
            None if arguments.no_calibration else split.iter_product_folds,
            key=key,
            seed=seed,
        )
        heads.meta.update({"layers": list(layers), "pool": arguments.pool, "apertus": arguments.model})
        digest = hashlib.sha256(key.encode()).hexdigest()[:8]
        saved["path"] = heads.save(out_dir() / "heads" / f"heads-{digest}.npz")
        return HeadsPredictor(heads, features)

    model_eval.PREDICTORS.setdefault("heads", factory)
    row = model_eval.run_eval(
        "heads", exp_id=arguments.exp_id, note=arguments.note or key, allow_dirty=arguments.allow_dirty
    )
    print(model_eval.format_row(row))
    print(f"heads saved to {saved.get('path')}; serve with {MODEL_ENV}=<that file> and "
          "--predictor galaxus.model.heads:predictor_from_env")
    return 0


if __name__ == "__main__":
    sys.exit(main())
