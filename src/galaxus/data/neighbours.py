"""Issue #112 Part A: a similar-image index over the prepared train images, as a development tool.

Why this part is allowed now: the written rules let any model be used during development, and the only use here is
finding suspected label errors - a second signal for #7's bonus csv, next to the model confidences of #75. Putting
neighbour labels into Apertus' prompt at runtime is **Part B**, which waits for the hosts' answer to Q-0003
(``decisions.md``) and belongs to the model lane. Nothing in this module runs at inference time and nothing here
imports a gateway client, so a "no" to Q-0003 leaves it exactly as it is.

What it does, in the order the command runs it:

1. Embed the **prepared** 512 px images - the pixels the model is actually served, through this lane's image cache,
   so an embedding and a prediction always describe the same picture.
2. Keep those embeddings as ``float16`` under ``$GALAXUS_DATA/cache/embeddings/<encoder>/``, one file per image, so
   an interrupted run resumes instead of starting over. Never in git (AGENTS.md product rule 3).
3. For each train image, find its ``k`` most similar train images by cosine similarity, **never one of its own
   product's** images: those share its labels, so they would agree with it trivially and prove nothing.
4. Where most of those neighbours carry a different human label, write the neighbours' majority label with their
   agreement as the confidence, in the shape :func:`galaxus.data.label_errors.find_label_errors` reads.
"""

from __future__ import annotations

import io
import json
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from PIL import Image as PillowImage

from galaxus.data.classes import ID_COLUMN, dimensions
from galaxus.data.images import IMAGE_SIZE, PATCH_SIZE, cached_jpeg
from galaxus.data.images import _id_path_pairs as id_path_pairs  # same lane, same owner; changes together
from galaxus.data.label_errors import _repository_holding
from galaxus.data.paths import cache_dir, out_dir
from galaxus.data.split import PRODUCT_COLUMN

#: The default encoder. Its revision and licence go in MODELS.md through the PR's "For design" note, not here:
#: AGENTS.md says never to write a fact like a licence from memory (product rule 4 and the "Hard rules" section).
DEFAULT_ENCODER = "google/siglip-base-patch16-224"
DEFAULT_K = 8
DEFAULT_MIN_SIMILARITY = 0.5
DEFAULT_PREDICTIONS_NAME = "neighbour_votes.jsonl"
#: Stored narrow on purpose: half the bytes, and cosine similarity does not need the extra precision.
EMBEDDING_DTYPE = np.float16


class Encoder(Protocol):
    """One method, so that a stub in a test is as real as the downloaded model as far as this module cares.

    ``encode`` takes prepared images and returns one row per image. Rows need not be unit length: every consumer
    here normalises, because cosine similarity is what the index is for.
    """

    def encode(self, images: Sequence[PillowImage.Image]) -> np.ndarray:  # pragma: no cover - a protocol
        ...


@dataclass(frozen=True)
class Index:
    """Embeddings to search, with the two id columns that decide what may match what.

    ``vectors`` is ``(n, d)`` and already unit length, so a dot product *is* the cosine similarity.
    ``products`` is the product of each row, which :func:`nearest` uses to refuse a row's own product.
    """

    image_ids: tuple[str, ...]
    products: tuple[str, ...]
    vectors: np.ndarray

    def __len__(self) -> int:
        return len(self.image_ids)


def encoder_slug(name: str = DEFAULT_ENCODER) -> str:
    """A folder-safe name for an encoder, so ``org/model`` does not become a directory level of its own."""
    safe = "".join(character if character.isalnum() or character in "-_." else "-" for character in name)
    return safe.strip("-") or "encoder"


def embedding_dir(encoder: str = DEFAULT_ENCODER) -> Path:
    """``$GALAXUS_DATA/cache/embeddings/<encoder>/``: one ``.npy`` per image, so a run resumes.

    The encoder is in the path because two encoders produce incomparable vectors; keeping them apart means a
    second encoder fills a new folder instead of silently mixing its numbers into the first one's.
    """
    return cache_dir("embeddings", encoder_slug(encoder))


def pooled_features(output: Any) -> Any:
    """The one image vector per row out of whatever an encoder returned.

    Needed because the shape is not stable across ``transformers`` versions: on 5.24 SigLIP's
    ``get_image_features`` returns a ``BaseModelOutputWithPooling`` (``last_hidden_state`` (n, patches, width) and
    ``pooler_output`` (n, width)), while older versions returned the pooled tensor directly. Calling ``.detach()``
    on the output object raises ``AttributeError``, so the unwrapping happens here, in a function tests can reach
    without downloading a model.

    ``pooler_output`` is preferred over a mean of ``last_hidden_state`` deliberately: a mean over the patch grid is
    permutation-invariant, so an image and its mirror come out bit-identical (measured on #127). The pooled head is
    not, and left-from-right is exactly what the perspective dimension needs. The mean is only a last resort for a
    model that offers nothing else.
    """
    for attribute in ("image_embeds", "pooler_output"):
        value = getattr(output, attribute, None)
        if value is not None:
            return value
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is not None:
        return hidden.mean(dim=1)
    return output


def load_encoder(name: str = DEFAULT_ENCODER) -> Encoder:
    """The default image encoder, imported only when it is actually wanted.

    ``transformers`` is in the ``local`` extra rather than the base dependencies, so importing it at module level
    would make this module unimportable for everyone who did not install that extra - including ``ci``. The error
    says which extra to install rather than letting an ImportError surface from three frames down.
    """
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as missing:  # pragma: no cover - depends on the installed extras
        raise ImportError(
            f"the default encoder needs the 'local' extra (torch and transformers): {missing}. "
            "Install it with `uv sync --extra local`, or pass an encoder of your own to embed_rows()"
        ) from missing

    processor = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.eval()

    class _Transformers:
        """Wraps the model so the rest of this module only sees :class:`Encoder`."""

        def encode(self, images: Sequence[PillowImage.Image]) -> np.ndarray:
            batch = processor(images=list(images), return_tensors="pt")
            with torch.no_grad():
                raw = model.get_image_features(**batch) if hasattr(model, "get_image_features") else model(**batch)
            return pooled_features(raw).detach().cpu().numpy()

    return _Transformers()


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    """``matrix`` with every row scaled to length 1, leaving an all-zero row alone instead of dividing by zero."""
    vectors = np.asarray(matrix, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"expected one row per image, got an array with shape {vectors.shape}")
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(lengths == 0, 1.0, lengths)


def embed_rows(
    rows: pd.DataFrame | Iterable[tuple[object, str | os.PathLike[str]]],
    encoder: Encoder,
    encoder_name: str = DEFAULT_ENCODER,
    batch_size: int = 16,
    image_size: int = IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
    directory: str | os.PathLike[str] | None = None,
    refresh: bool = False,
) -> dict[str, object]:
    """Embed every row once into the embedding cache, resumably.

    The pixels come from this lane's image cache (:func:`galaxus.data.images.cached_jpeg`), so they are the
    prepared 512 px images rather than the originals: an embedding therefore describes the same picture the model
    is served, which is the only way a neighbour's label says anything about this image's label.

    Returns:
        How many embeddings were ``written``, ``skipped`` (already cached) and ``missing`` (no source image), and
        the ``directory`` they are in. Counts only - no image id - so the result is safe to print.
    """
    folder = Path(directory) if directory is not None else embedding_dir(encoder_name)
    folder.mkdir(parents=True, exist_ok=True)
    pairs = id_path_pairs(rows)

    todo = [pair for pair in pairs if refresh or not (folder / f"{pair[0]}.npy").is_file()]
    written = missing = 0

    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        images: list[PillowImage.Image] = []
        kept: list[object] = []
        for image_id, image_file in batch:
            try:
                data = cached_jpeg(image_id, image_file, image_size=image_size, patch_size=patch_size)
            except FileNotFoundError:
                missing += 1
                continue
            images.append(PillowImage.open(io.BytesIO(data)).convert("RGB"))
            kept.append(image_id)
        if not images:
            continue
        vectors = np.asarray(encoder.encode(images))
        if len(vectors) != len(kept):
            raise ValueError(f"the encoder returned {len(vectors)} row(s) for {len(kept)} image(s)")
        for image_id, vector in zip(kept, vectors, strict=True):
            target = folder / f"{image_id}.npy"
            partial = target.with_name(target.name + ".part")
            # Written to a .part name and renamed, like the image cache, so an interrupted run never leaves a
            # half-written .npy that the next run would mistake for a finished one.
            # Written through an open handle, not by name: np.save appends ".npy" to a name that lacks it, so
            # saving to "<id>.npy.part" by name would quietly produce "<id>.npy.part.npy" and leave nothing to
            # rename. Caught by the resume test rather than by reading the docs.
            with partial.open("wb") as handle:
                np.save(handle, np.asarray(vector, dtype=EMBEDDING_DTYPE))
            partial.replace(target)
            written += 1

    return {
        "written": written,
        "skipped": len(pairs) - len(todo),
        "missing": missing,
        "directory": str(folder),
    }


def build_index(
    train: pd.DataFrame,
    encoder_name: str = DEFAULT_ENCODER,
    directory: str | os.PathLike[str] | None = None,
    exclude_ids: Iterable[object] = (),
) -> Index:
    """An :class:`Index` of the cached embeddings of ``train``'s rows, in ``train``'s order.

    Args:
        train: rows with an image id and a ``product_id``. Only train rows belong in an index: dev and final
            images are never in it, because a hint drawn from them would be labelling them by hand.
        exclude_ids: image ids to leave out. Pass the holdout's ids when the index serves a holdout evaluation,
            or a neighbour of a holdout image could carry a label the model is about to be scored on.

    Raises:
        KeyError: if ``product_id`` is missing, since without it a row's own product cannot be refused.
        FileNotFoundError: if no row has an embedding yet, naming the command that fills them.
    """
    if PRODUCT_COLUMN not in train.columns:
        raise KeyError(
            f"{PRODUCT_COLUMN} is not a column of this table: without it the index cannot refuse a row's own "
            "product, and a product's other images share its labels"
        )
    folder = Path(directory) if directory is not None else embedding_dir(encoder_name)
    dropped = {str(image_id) for image_id in exclude_ids}

    image_ids: list[str] = []
    products: list[str] = []
    vectors: list[np.ndarray] = []
    for image_id, product in zip(train[ID_COLUMN].astype(str), train[PRODUCT_COLUMN].astype(str), strict=True):
        if image_id in dropped:
            continue
        path = folder / f"{image_id}.npy"
        if not path.is_file():
            continue
        image_ids.append(image_id)
        products.append(product)
        vectors.append(np.load(path))

    if not vectors:
        raise FileNotFoundError(
            f"no embedding found in {folder} for any of these {len(train)} row(s). Fill them first with "
            "`python -m galaxus.data.neighbours embed`"
        )
    return Index(tuple(image_ids), tuple(products), unit_rows(np.vstack(vectors)))


def nearest(
    queries: Index,
    index: Index,
    k: int = DEFAULT_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    chunk: int = 512,
) -> dict[str, list[tuple[str, float]]]:
    """The ``k`` most similar index rows per query, closest first, by cosine similarity.

    A row's own product is never a neighbour, and neither is the row itself: its product's other images share its
    labels, so they would agree with it whatever the label is, and an index that returns them would report
    agreement it has not earned.

    Args:
        k: how many neighbours to keep per query, after the exclusions.
        min_similarity: neighbours below this are dropped, so a query in a thin part of the space gets few
            neighbours or none rather than distant ones.
        chunk: how many queries to score at once, to keep the similarity matrix small on a large index.

    Returns:
        ``{query image id: [(neighbour image id, similarity), ...]}``, without the queries that found nobody.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1 to have a neighbour, got {k}")
    if not len(index):
        raise ValueError("the index is empty, so nothing can be a neighbour")

    index_products = np.asarray(index.products)
    index_ids = np.asarray(index.image_ids)
    found: dict[str, list[tuple[str, float]]] = {}

    for start in range(0, len(queries), chunk):
        stop = min(start + chunk, len(queries))
        similarity = queries.vectors[start:stop] @ index.vectors.T
        for offset in range(stop - start):
            position = start + offset
            row = similarity[offset].copy()
            # The two exclusions, as a mask rather than a filter afterwards, so `k` counts eligible neighbours.
            row[index_products == queries.products[position]] = -np.inf
            row[index_ids == queries.image_ids[position]] = -np.inf

            eligible = int(np.sum(np.isfinite(row) & (row >= min_similarity)))
            if not eligible:
                continue
            take = min(k, eligible)
            best = np.argpartition(-row, take - 1)[:take]
            best = best[np.argsort(-row[best], kind="stable")]
            found[queries.image_ids[position]] = [(str(index_ids[j]), float(row[j])) for j in best]

    return found


def neighbour_votes(
    train: pd.DataFrame,
    neighbours: dict[str, list[tuple[str, float]]],
    dimension_names: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """What the neighbours say each image's label should be, with their agreement as the confidence.

    Only a neighbour's **human** label counts: a model prelabel is the thing being checked, so letting it vote
    would have the pipeline confirm itself. A dimension no neighbour has a human label for is left out of the
    record rather than guessed at.

    Returns:
        One record per image, in the shape :func:`galaxus.data.label_errors.out_of_fold_frame` reads:
        ``{"image_id": ..., "labels": {dimension: majority}, "confidence": {dimension: agreement}}``.
    """
    names = tuple(dimension_names) if dimension_names is not None else dimensions()
    indexed = train.set_index(train[ID_COLUMN].astype(str))

    records: list[dict[str, Any]] = []
    for image_id, close in neighbours.items():
        labels: dict[str, str] = {}
        confidence: dict[str, float] = {}
        for dimension in names:
            source = f"{dimension}_source"
            if dimension not in indexed.columns or source not in indexed.columns:
                raise KeyError(f"train.csv has no {dimension}/{source} column: neighbour labels cannot be read")
            votes: dict[str, int] = {}
            for neighbour_id, _ in close:
                if neighbour_id not in indexed.index:
                    continue
                row = indexed.loc[neighbour_id]
                if str(row[source]) != "human":
                    continue
                value = str(row[dimension])
                if value == "":
                    continue
                votes[value] = votes.get(value, 0) + 1
            total = sum(votes.values())
            if not total:
                continue
            # Ties go to the lexicographically smaller class, so the same index gives the same file every run.
            winner = min(votes.items(), key=lambda item: (-item[1], item[0]))[0]
            labels[dimension] = winner
            confidence[dimension] = votes[winner] / total
        if labels:
            records.append({ID_COLUMN: image_id, "labels": labels, "confidence": confidence})
    records.sort(key=lambda record: str(record[ID_COLUMN]))
    return records


def write_votes(
    records: Sequence[dict[str, Any]],
    path: str | os.PathLike[str] | None = None,
    name: str = DEFAULT_PREDICTIONS_NAME,
) -> Path:
    """Write ``records`` as the jsonl ``find_label_errors`` reads, refusing anywhere inside a git work tree.

    These records hold train labels - a neighbour's human label is a competition label - so they fall under
    AGENTS.md product rule 3 exactly as the label-error csv does. The guard is here and not only in the command
    because a caller that passes a path of its own has to be refused too.
    """
    target = (Path(path) if path is not None else out_dir() / name).expanduser()
    target = target.resolve() if target.is_absolute() else (Path.cwd() / target).resolve()
    repository = _repository_holding(target)
    if repository is not None:
        raise ValueError(
            f"{target} is inside the git work tree at {repository}, and these records hold train labels, which "
            "never enter git (AGENTS.md product rule 3). Write them under $GALAXUS_DATA/out/ instead, which is "
            "where they go by default"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return target


def main(argv: list[str] | None = None) -> int:
    """``python -m galaxus.data.neighbours``: fill the embedding cache, or write the neighbour votes (#112).

    Prints counts only - never an image id (product rule 3) - and writes the votes under ``$GALAXUS_DATA/out/``,
    never into the repository.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m galaxus.data.neighbours",
        description="A similar-image index over the prepared train images, as a development tool (#112 part A).",
    )
    parser.add_argument("--encoder", default=DEFAULT_ENCODER, help=f"encoder name (default {DEFAULT_ENCODER})")
    commands = parser.add_subparsers(dest="command", required=True)

    embed = commands.add_parser("embed", help="fill the embedding cache for train.csv, resumably")
    embed.add_argument("--batch-size", type=int, default=16)

    errors = commands.add_parser("label-errors", help="write the neighbour votes for find_label_errors to read")
    errors.add_argument("--k", type=int, default=DEFAULT_K, help=f"neighbours per image (default {DEFAULT_K})")
    errors.add_argument(
        "--min-similarity",
        type=float,
        default=DEFAULT_MIN_SIMILARITY,
        help=f"ignore neighbours below this cosine similarity (default {DEFAULT_MIN_SIMILARITY})",
    )
    errors.add_argument("--dimensions", help="comma-separated subset of dimensions (default: all of them)")
    errors.add_argument("--out", help=f"where to write (default $GALAXUS_DATA/out/{DEFAULT_PREDICTIONS_NAME})")

    arguments = parser.parse_args(argv)
    if arguments.command == "label-errors" and arguments.k < 1:
        parser.error("--k must be at least 1")

    from galaxus.data.load import read_train

    try:
        train = read_train()
        if arguments.command == "embed":
            counts = embed_rows(
                train, load_encoder(arguments.encoder), arguments.encoder, batch_size=arguments.batch_size
            )
            print(
                f"[neighbours] {counts['written']} embedded, {counts['skipped']} already cached, "
                f"{counts['missing']} without a source image, in {counts['directory']}"
            )
            return 0

        names = tuple(name.strip() for name in arguments.dimensions.split(",")) if arguments.dimensions else None
        index = build_index(train, arguments.encoder)
        close = nearest(index, index, k=arguments.k, min_similarity=arguments.min_similarity)
        records = neighbour_votes(train, close, names)
        written = write_votes(records, arguments.out)
    except (OSError, KeyError, ValueError, ImportError) as failure:
        print(f"cannot work out the neighbours: {failure}", file=sys.stderr)
        return 1

    print(
        f"[neighbours] {len(index)} image(s) in the index, {len(close)} with at least one neighbour at "
        f"similarity >= {arguments.min_similarity}, {len(records)} with a neighbour label to compare"
    )
    print(f"[neighbours] votes written to {written} - holds train labels, so never commit it")
    print(f"[neighbours] next: python -m galaxus.data.label_errors --predictions {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
