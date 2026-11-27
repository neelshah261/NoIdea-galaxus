"""One interface to Apertus v1.5, whatever runs it (issue #13, BASE_ARCHITECTURE B6).

- ``generate(messages, images, json_schema)`` on every backend: chat messages and images in, one answer out;
- ``forward(images, prompt)`` on ``local`` only: hidden states, for heads on frozen features (Q-0007).

Images given to a backend are already prepared: transparency composited onto white and resized by the data lane
(``galaxus.data.images.prepared_jpeg`` or ``prepare_image``). A backend never resizes and refuses a transparent
image, because a plain ``convert("RGB")`` turns transparency black (FAQ.md, "Transparent PNGs turn black").
"""

from __future__ import annotations

import base64
import io
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from PIL import Image as PillowImage

#: The Apertus v1.5 models on Hugging Face, as README.md "Resources & Tools" lists them.
APERTUS_8B = "swiss-ai/Apertus-v1.5-8B"
APERTUS_70B = "swiss-ai/Apertus-v1.5-70B"
#: Room for a six-dimension JSON answer. The Galaxus serving setup caps answers at about 120 tokens (#17).
DEFAULT_MAX_TOKENS = 256
#: JPEG quality for a PIL image we encode ourselves, the data lane's and the reference notebook's.
JPEG_QUALITY = 92

#: A PIL image, encoded JPEG / PNG / WebP bytes (for example ``prepared_jpeg``), or the path of such a file.
ImageInput = PillowImage.Image | bytes | str | os.PathLike
#: A chat message: ``{"role": "system" | "user" | "assistant", "content": "<text>"}``.
Message = Mapping[str, Any]


class BackendError(RuntimeError):
    """A backend could not answer: the server refused, the model did not load, a setting is missing."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Generation:
    """One answer. Token counts are the backend's own, None where it reports none."""

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    model: str | None = None
    seconds: float | None = None


@runtime_checkable
class Backend(Protocol):
    """What every backend offers. ``local`` also has ``forward``."""

    name: str
    model: str

    def generate(
        self,
        messages: Sequence[Message],
        images: Sequence[ImageInput] = (),
        json_schema: Mapping[str, Any] | None = None,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> Generation:
        """Answer ``messages``; ``images`` go with the last user message, in order, before its text."""
        ...


def _is_transparent(image: PillowImage.Image) -> bool:
    return image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info


def _refuse_transparent(image: PillowImage.Image) -> None:
    if _is_transparent(image):
        raise ValueError(
            "a backend gets prepared images only: composite transparency onto white first "
            "(galaxus.data.images.opaque_rgb_on_white), a plain convert('RGB') turns it black"
        )


def _read(image: bytes | str | os.PathLike) -> bytes:
    return image if isinstance(image, bytes) else Path(image).read_bytes()


def image_mime(data: bytes) -> str:
    """The MIME type of encoded image bytes."""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("an image must be JPEG, PNG or WebP bytes, the path of such a file, or a PIL image")


def image_bytes(image: ImageInput) -> tuple[bytes, str]:
    """``image`` as encoded bytes and their MIME type. Bytes and files pass unchanged; a PIL image becomes JPEG."""
    if isinstance(image, PillowImage.Image):
        _refuse_transparent(image)
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=JPEG_QUALITY)
        return buffer.getvalue(), "image/jpeg"
    data = _read(image)
    return data, image_mime(data)


def data_url(image: ImageInput) -> str:
    """``image`` as a base64 ``data:`` URL, the form OpenAI-compatible servers take images in."""
    data, mime = image_bytes(image)
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def pil_image(image: ImageInput) -> PillowImage.Image:
    """``image`` decoded as an RGB PIL image, for backends that run the model in this process."""
    if isinstance(image, PillowImage.Image):
        _refuse_transparent(image)
        return image.convert("RGB")
    with PillowImage.open(io.BytesIO(_read(image))) as opened:
        _refuse_transparent(opened)
        return opened.convert("RGB")


def last_user_index(messages: Sequence[Message]) -> int:
    """Where the images go: the last user message."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    raise ValueError("images need a user message to go with")
