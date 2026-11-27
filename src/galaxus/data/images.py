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
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from PIL import Image as PillowImage
from PIL import ImageOps

from galaxus.data.classes import ID_COLUMN
from galaxus.data.load import IMAGE_FILE_COLUMN
from galaxus.data.paths import cache_dir

#: Vision patch size: both sides of a prepared image are a multiple of it.
PATCH_SIZE = 32
#: An image is never scaled so that its longest side exceeds ``MAX_SIDE_FACTOR * image_size``.
MAX_SIDE_FACTOR = 8
#: Default target area, about 512 x 512 pixels, as in the reference notebook. We ablate 384 / 512 / 768.
IMAGE_SIZE = 512
#: JPEG quality of the encoded image, as in the reference notebook.
JPEG_QUALITY = 92


def opaque_rgb_on_white(image: PillowImage.Image) -> PillowImage.Image:
    """RGB, with any transparency composited onto white."""
    if image.mode not in ("RGBA", "LA", "PA") and "transparency" not in image.info:
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    canvas = PillowImage.new("RGBA", rgba.size, (255, 255, 255, 255))
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB")


def native_aspect_image(image: PillowImage.Image, image_size: int = IMAGE_SIZE) -> PillowImage.Image:
    """The image scaled to about ``image_size`` squared pixels, aspect ratio kept.

    Each side is a multiple of :data:`PATCH_SIZE`.
    """
    scale = image_size / (image.width * image.height) ** 0.5
    scale *= min(1.0, MAX_SIDE_FACTOR * image_size / (max(image.width, image.height) * scale))
    width = max(PATCH_SIZE, round(image.width * scale / PATCH_SIZE) * PATCH_SIZE)
    height = max(PATCH_SIZE, round(image.height * scale / PATCH_SIZE) * PATCH_SIZE)
    return image.resize((width, height), PillowImage.LANCZOS)


def prepared_jpeg(
    image_file: str | os.PathLike[str] | io.IOBase,
    mirrored: bool = False,
    image_size: int = IMAGE_SIZE,
    jpeg_quality: int = JPEG_QUALITY,
) -> bytes:
    """The image as the model gets it, as JPEG bytes, mirrored first if asked."""
    _check_image_size(image_size)
    with PillowImage.open(image_file) as image:
        rgb = opaque_rgb_on_white(image)
    if mirrored:
        rgb = ImageOps.mirror(rgb)
    buffer = io.BytesIO()
    native_aspect_image(rgb, image_size).save(buffer, format="JPEG", quality=jpeg_quality)
    return buffer.getvalue()


def prepare_image(
    image_file: str | os.PathLike[str] | io.IOBase,
    mirrored: bool = False,
    image_size: int = IMAGE_SIZE,
    jpeg_quality: int = JPEG_QUALITY,
) -> PillowImage.Image:
    """The prepared JPEG decoded again: exactly the pixels the model is served."""
    return PillowImage.open(io.BytesIO(prepared_jpeg(image_file, mirrored, image_size, jpeg_quality))).convert("RGB")


def vision_tokens(image: PillowImage.Image) -> int:
    """The vision-token count of a prepared image. Any other count means something resized it."""
    return (image.width // PATCH_SIZE) * (image.height // PATCH_SIZE)


def image_cache_dir(image_size: int = IMAGE_SIZE) -> Path:
    """``$GALAXUS_DATA/cache/images/<size>/``: one prepared JPEG per image, as the data layout specifies."""
    _check_image_size(image_size)
    return cache_dir("images", str(image_size))


def _cache_name(image_id: object, mirrored: bool) -> str:
    """The cache file name for an image id, refusing anything that is not a plain name.

    The ids come from a competition CSV, so a value carrying a separator or ``..`` must never be able to write
    outside the cache directory.
    """
    name = str(image_id)
    if not name or name in {".", ".."} or "/" in name or "\\" in name or Path(name).name != name:
        raise ValueError(f"image_id {image_id!r} is not a plain file name, so it cannot be used as a cache key")
    return f"{name}.mirrored.jpg" if mirrored else f"{name}.jpg"


def cached_jpeg(
    image_id: object,
    image_file: str | os.PathLike[str],
    mirrored: bool = False,
    image_size: int = IMAGE_SIZE,
    jpeg_quality: int = JPEG_QUALITY,
    refresh: bool = False,
    directory: str | os.PathLike[str] | None = None,
) -> bytes:
    """:func:`prepared_jpeg`, but written through the image cache so the same pixels are prepared once.

    The mirror is cached under its own name, so test-time augmentation does not evict the original.
    """
    folder = Path(directory) if directory is not None else image_cache_dir(image_size)
    target = folder / _cache_name(image_id, mirrored)
    if target.is_file() and not refresh:
        return target.read_bytes()
    data = prepared_jpeg(image_file, mirrored=mirrored, image_size=image_size, jpeg_quality=jpeg_quality)
    folder.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(data)
    partial.replace(target)
    return data


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
) -> dict[str, object]:
    """Prepare every image once into the cache; this is what ``just prep`` fills (BASE_ARCHITECTURE.md B1).

    Safe to re-run and safe to interrupt: an image already cached is skipped, and each file is written to a
    ``.part`` name and renamed, so a half-written JPEG is never mistaken for a cached one.

    Returns:
        How many files were ``written``, ``skipped`` and ``missing``, and the ``directory`` they are in.
    """
    folder = Path(directory) if directory is not None else image_cache_dir(image_size)
    written = skipped = missing = 0
    for image_id, image_file in _id_path_pairs(rows):
        if (folder / _cache_name(image_id, mirrored)).is_file() and not refresh:
            skipped += 1
            continue
        try:
            cached_jpeg(
                image_id,
                image_file,
                mirrored=mirrored,
                image_size=image_size,
                jpeg_quality=jpeg_quality,
                refresh=refresh,
                directory=folder,
            )
        except FileNotFoundError:
            if not skip_missing:
                raise
            missing += 1
            continue
        written += 1
    return {"written": written, "skipped": skipped, "missing": missing, "directory": folder}


def _check_image_size(image_size: int) -> None:
    if not isinstance(image_size, int) or image_size < PATCH_SIZE or image_size % PATCH_SIZE:
        raise ValueError(f"image_size must be a multiple of {PATCH_SIZE} and at least {PATCH_SIZE}, got {image_size!r}")
