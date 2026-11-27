"""Turning an image file into the pixels the model sees. One function for training and for prediction.

The three steps are the ones our labelers and the Qwen baseline saw, so our score is comparable to the baselines in
[challenge_description.md, "Baselines"](../../../challenge_description.md#baselines):

1. transparency composited onto **white**, not black: see [FAQ.md, "Transparent PNGs turn
   black"](../../../FAQ.md#data-and-labels) and the ``opaque_rgb_on_white`` cell of
   [reference/finetune_qwen35_reference.ipynb](../../../reference/finetune_qwen35_reference.ipynb), section "Images";
2. an optional horizontal mirror, for augmentation and test-time augmentation;
3. a resize to about ``image_size``-squared pixels that keeps the aspect ratio and makes both sides a multiple of
   ``PATCH_SIZE``, so the vision-token count is ``(width / PATCH_SIZE) * (height / PATCH_SIZE)``.

``opaque_rgb_on_white`` and ``native_aspect_image`` keep the reference notebook's behaviour exactly; changing them
changes what every lane sees.
"""

from __future__ import annotations

import io
import os
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image as PillowImage
from PIL import ImageOps

from galaxus.data.classes import ID_COLUMN
from galaxus.data.load import IMAGE_FILE_COLUMN
from galaxus.data.paths import cache_dir

#: Default vision patch size: both sides of a prepared image are a multiple of it.
#:
#: 32 is **Qwen's**, from the reference notebook, not Apertus v1.5's. Apertus' is Q-0002, measured as pixels per
#: vision token by the B6 smoke test (#13, #46). Since #28 every function here takes ``patch_size``, so answering
#: Q-0002 is a change to this one line rather than an edit through the code; the cache folder carries the patch
#: size (``<size>-p<patch>``), so the old pixels are never served to a model expecting the new ones.
#: TODO(owner): set this from the #46 measurement and say so on Q-0002.
PATCH_SIZE = 32
#: An image is never scaled so that its longest side exceeds ``MAX_SIDE_FACTOR * image_size``.
MAX_SIDE_FACTOR = 8
#: Default target area, about 512 x 512 pixels, as in the reference notebook. We ablate 384 / 512 / 768.
IMAGE_SIZE = 512
#: JPEG quality of the encoded image, as in the reference notebook.
JPEG_QUALITY = 92

#: What marks the mirrored copy in the cache: ``<image_id>.mirror.jpg`` (issue #27).
MIRROR_SUFFIX = ".mirror"


def opaque_rgb_on_white(image: PillowImage.Image) -> PillowImage.Image:
    """RGB, with any transparency composited onto white."""
    if image.mode not in ("RGBA", "LA", "PA") and "transparency" not in image.info:
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    canvas = PillowImage.new("RGBA", rgba.size, (255, 255, 255, 255))
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB")


def native_aspect_image(
    image: PillowImage.Image, image_size: int = IMAGE_SIZE, patch_size: int = PATCH_SIZE
) -> PillowImage.Image:
    """The image scaled to about ``image_size`` squared pixels, aspect ratio kept.

    Each side is a multiple of ``patch_size``, because the model sees whole patches.
    """
    _check_patch_size(patch_size)
    scale = image_size / (image.width * image.height) ** 0.5
    scale *= min(1.0, MAX_SIDE_FACTOR * image_size / (max(image.width, image.height) * scale))
    width = max(patch_size, round(image.width * scale / patch_size) * patch_size)
    height = max(patch_size, round(image.height * scale / patch_size) * patch_size)
    return image.resize((width, height), PillowImage.LANCZOS)


def prepared_jpeg(
    image_file: str | os.PathLike[str] | io.IOBase,
    mirrored: bool = False,
    image_size: int = IMAGE_SIZE,
    jpeg_quality: int = JPEG_QUALITY,
    patch_size: int = PATCH_SIZE,
) -> bytes:
    """The image as the model gets it, as JPEG bytes, mirrored first if asked."""
    _check_image_size(image_size, patch_size)
    with PillowImage.open(image_file) as image:
        rgb = opaque_rgb_on_white(image)
    if mirrored:
        rgb = ImageOps.mirror(rgb)
    buffer = io.BytesIO()
    native_aspect_image(rgb, image_size, patch_size).save(buffer, format="JPEG", quality=jpeg_quality)
    return buffer.getvalue()


def prepare_image(
    image_file: str | os.PathLike[str] | io.IOBase,
    mirrored: bool = False,
    image_size: int = IMAGE_SIZE,
    jpeg_quality: int = JPEG_QUALITY,
    patch_size: int = PATCH_SIZE,
) -> PillowImage.Image:
    """The prepared JPEG decoded again: exactly the pixels the model is served."""
    prepared = prepared_jpeg(image_file, mirrored, image_size, jpeg_quality, patch_size)
    return PillowImage.open(io.BytesIO(prepared)).convert("RGB")


def vision_tokens(image: PillowImage.Image, patch_size: int = PATCH_SIZE) -> int:
    """The vision-token count of a prepared image. Any other count means something resized it."""
    _check_patch_size(patch_size)
    return (image.width // patch_size) * (image.height // patch_size)


def image_cache_dir(image_size: int = IMAGE_SIZE, patch_size: int = PATCH_SIZE) -> Path:
    """``$GALAXUS_DATA/cache/images/<size>-p<patch>/``: one prepared JPEG per image (issue #27).

    The patch size is in the folder name, not only the image size, because it decides the pixels: the prepared
    image is rounded to whole patches, so the same ``image_size`` at a different patch size is a different
    picture. Keeping it in the name means a future patch size (#28) fills a new folder instead of silently
    serving the old one's pixels.

    Args:
        image_size: the target area per side, a multiple of ``patch_size``.
        patch_size: the patch size these pixels were prepared at. Since #28 this is a real parameter, so two
            patch sizes keep two caches and can never serve each other's pixels.
    """
    _check_image_size(image_size, patch_size)
    return cache_dir("images", f"{image_size}-p{patch_size}")


def _cache_name(image_id: object, mirrored: bool) -> str:
    """The cache file name for an image id, refusing anything that is not a plain name.

    The ids come from a competition CSV, so a value carrying a separator or ``..`` must never be able to write
    outside the cache directory.
    """
    name = str(image_id)
    if not name or name in {".", ".."} or "/" in name or "\\" in name or Path(name).name != name:
        raise ValueError(f"image_id {image_id!r} is not a plain file name, so it cannot be used as a cache key")
    return f"{name}{MIRROR_SUFFIX}.jpg" if mirrored else f"{name}.jpg"


def cached_jpeg(
    image_id: object,
    image_file: str | os.PathLike[str],
    mirrored: bool = False,
    image_size: int = IMAGE_SIZE,
    jpeg_quality: int = JPEG_QUALITY,
    refresh: bool = False,
    directory: str | os.PathLike[str] | None = None,
    patch_size: int = PATCH_SIZE,
) -> bytes:
    """:func:`prepared_jpeg`, but written through the image cache so the same pixels are prepared once.

    The mirror is cached under its own name, so test-time augmentation does not evict the original.
    """
    folder = Path(directory) if directory is not None else image_cache_dir(image_size, patch_size)
    target = folder / _cache_name(image_id, mirrored)
    if target.is_file() and not refresh:
        return target.read_bytes()
    data = prepared_jpeg(
        image_file, mirrored=mirrored, image_size=image_size, jpeg_quality=jpeg_quality, patch_size=patch_size
    )
    folder.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(data)
    partial.replace(target)
    return data


_SOURCE_INDEX: dict[str, dict[str, str]] = {}


def _source_index(refresh: bool = False) -> dict[str, str]:
    """``image_id`` → source image path, built once per ``$GALAXUS_DATA`` from whichever parts are on disk.

    Only used when a caller asks for a cached image by id alone. A part that is not there yet (``final.csv``
    before Saturday) is simply skipped rather than raising.
    """
    from galaxus.data.paths import data_dir

    key = str(data_dir())
    if refresh:
        _SOURCE_INDEX.pop(key, None)
    if key not in _SOURCE_INDEX:
        from galaxus.data import load

        index: dict[str, str] = {}
        for reader in (load.read_train, load.read_dev, load.read_final):
            try:
                frame = reader()
            except (FileNotFoundError, KeyError, ValueError):
                continue
            if ID_COLUMN in frame.columns and IMAGE_FILE_COLUMN in frame.columns:
                index.update(zip(frame[ID_COLUMN].astype(str), frame[IMAGE_FILE_COLUMN].astype(str), strict=True))
        _SOURCE_INDEX[key] = index
    return _SOURCE_INDEX[key]


def cached_image(
    image_id: object,
    image_file: str | os.PathLike[str] | None = None,
    mirrored: bool = False,
    image_size: int = IMAGE_SIZE,
    jpeg_quality: int = JPEG_QUALITY,
    directory: str | os.PathLike[str] | None = None,
    patch_size: int = PATCH_SIZE,
) -> Path:
    """The **path** of one prepared image in the cache, preparing it first if it is not there yet (issue #27).

    This is what another lane should call: it hands back a file on disk, so a trainer or a runner can open it
    without going through this module. :func:`cached_jpeg` returns the bytes instead.

    Args:
        image_file: where to prepare it from. Left out, the id is looked up in the parts on disk, so a caller
            holding only an ``image_id`` still works.

    Raises:
        KeyError: the id is not in any part on disk, so there is nothing to prepare it from.
    """
    folder = Path(directory) if directory is not None else image_cache_dir(image_size, patch_size)
    target = folder / _cache_name(image_id, mirrored)
    if target.is_file():
        return target
    source = image_file
    if source is None:
        found = _source_index().get(str(image_id))
        if found is None:
            raise KeyError(
                f"image_id {image_id!r} is in no part on disk, so it cannot be prepared; "
                "pass image_file, or check $GALAXUS_DATA"
            )
        source = found
    cached_jpeg(
        image_id,
        source,
        mirrored=mirrored,
        image_size=image_size,
        jpeg_quality=jpeg_quality,
        directory=folder,
        patch_size=patch_size,
    )
    return target


def _id_path_pairs(rows: pd.DataFrame | Iterable[tuple[object, str | os.PathLike[str]]]) -> list[tuple]:
    """``(image_id, image_file)`` pairs from a table read by :func:`galaxus.data.load.read_part`, or from pairs."""
    if isinstance(rows, pd.DataFrame):
        missing = [column for column in (ID_COLUMN, IMAGE_FILE_COLUMN) if column not in rows.columns]
        if missing:
            raise KeyError(f"the table is missing {missing}; read it with galaxus.data.load.read_part")
        return list(zip(rows[ID_COLUMN], rows[IMAGE_FILE_COLUMN], strict=True))
    return [tuple(pair) for pair in rows]


def fill_image_cache(
    rows: pd.DataFrame | Iterable[tuple[object, str | os.PathLike[str]]],
    mirrored: bool = False,
    image_size: int = IMAGE_SIZE,
    jpeg_quality: int = JPEG_QUALITY,
    refresh: bool = False,
    directory: str | os.PathLike[str] | None = None,
    skip_missing: bool = False,
    workers: int = 1,
    patch_size: int = PATCH_SIZE,
) -> dict[str, object]:
    """Prepare every image once into the cache; this is what ``just prep`` fills (BASE_ARCHITECTURE.md B1).

    Safe to re-run and safe to interrupt: an image already cached is skipped, and each file is written to a
    ``.part`` name and renamed, so a half-written JPEG is never mistaken for a cached one.

    Returns:
        How many files were ``written``, ``skipped`` and ``missing``, and the ``directory`` they are in.
    """
    folder = Path(directory) if directory is not None else image_cache_dir(image_size, patch_size)
    pairs = _id_path_pairs(rows)
    todo = [
        (image_id, image_file)
        for image_id, image_file in pairs
        if refresh or not (folder / _cache_name(image_id, mirrored)).is_file()
    ]
    skipped = len(pairs) - len(todo)
    written = missing = 0
    started = time.perf_counter()

    def prepare_one(pair: tuple[object, str | os.PathLike[str]]) -> bool:
        """True when it was written, False when the source image was missing and that is allowed."""
        image_id, image_file = pair
        try:
            cached_jpeg(
                image_id,
                image_file,
                mirrored=mirrored,
                image_size=image_size,
                jpeg_quality=jpeg_quality,
                refresh=refresh,
                directory=folder,
                patch_size=patch_size,
            )
        except FileNotFoundError:
            if not skip_missing:
                raise
            return False
        return True

    if workers > 1 and len(todo) > 1:
        # Threads, not processes: the work is reading, JPEG decoding and writing, and Pillow drops the GIL for
        # the codecs. Each image is independent and written through its own ``.part`` file, so there is nothing
        # to lock. An exception in a worker still propagates out of the ``with``.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for was_written in pool.map(prepare_one, todo):
                written += was_written
                missing += not was_written
    else:
        for pair in todo:
            was_written = prepare_one(pair)
            written += was_written
            missing += not was_written

    seconds = time.perf_counter() - started
    return {
        "written": written,
        "skipped": skipped,
        "missing": missing,
        "directory": folder,
        "seconds": seconds,
        "images_per_second": written / seconds if written and seconds > 0 else None,
        "workers": workers,
    }


def _check_patch_size(patch_size: int) -> None:
    if not isinstance(patch_size, int) or isinstance(patch_size, bool) or patch_size < 1:
        raise ValueError(f"patch_size must be a positive whole number of pixels, got {patch_size!r}")


def _check_image_size(image_size: int, patch_size: int = PATCH_SIZE) -> None:
    _check_patch_size(patch_size)
    if not isinstance(image_size, int) or image_size < patch_size or image_size % patch_size:
        raise ValueError(
            f"image_size must be a multiple of the patch size {patch_size} and at least {patch_size}, "
            f"got {image_size!r}"
        )
