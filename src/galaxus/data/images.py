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

from PIL import Image as PillowImage
from PIL import ImageOps

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


def _check_image_size(image_size: int) -> None:
    if not isinstance(image_size, int) or image_size < PATCH_SIZE or image_size % PATCH_SIZE:
        raise ValueError(f"image_size must be a multiple of {PATCH_SIZE} and at least {PATCH_SIZE}, got {image_size!r}")
