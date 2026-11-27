"""Smoke test of one Apertus backend, with images drawn here: no competition data (issue #13).

    uv run python -m galaxus.model.apertus.smoke --backend api
    uv run python -m galaxus.model.apertus.smoke --backend vllm --base-url http://localhost:8000/v1
    uv run --with torch --with transformers python -m galaxus.model.apertus.smoke --backend local

It answers B6's open questions for the backend it runs on and prints them as a "For design" block for the PR:

- Q-0001: does the backend take an image, and does the model see it (a red square on white: does it say red)?
- Q-0002: is a JSON schema enforced (asked to answer "hello", does it still write schema JSON), and how many
  prompt tokens does an image cost at a few sizes, which gives the pixels per vision token the data lane's resize
  should use;
- Q-0006 (local only): which transformers version and model class loaded Apertus, and do hidden states come out;
- #75: does an answer held to the schema come with token logprobs (the zero-shot confidences need them)?

About a dozen short requests in all. The API key is never printed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image as PillowImage
from PIL import ImageDraw

from galaxus.model.apertus.backend import APERTUS_8B, Backend, BackendError, Generation
from galaxus.model.apertus.vllm import DEFAULT_BASE_URL

PROBE = "What color is the square in the image? Answer with one word."
#: Square sizes and one wide image: tokens that grow with the area mean the server keeps our resolution.
SIZES = ((256, 256), (512, 512), (768, 768), (512, 256))
SCHEMA = {
    "type": "object",
    "properties": {
        "color": {"type": "string", "enum": ["red", "green", "blue"]},
        "shape": {"type": "string", "enum": ["square", "circle", "triangle"]},
    },
    "required": ["color", "shape"],
    "additionalProperties": False,
}
#: Asks for something the schema forbids: schema JSON in the answer means the server enforced the schema.
SCHEMA_PROBE = "Reply with only the word hello. Do not write JSON."
THINK_MARKERS = ("<think>", "</think>")


def square(size: tuple[int, int] = (512, 512), color: tuple[int, int, int] = (220, 30, 30)) -> PillowImage.Image:
    """A colored square in the middle of a white image."""
    width, height = size
    image = PillowImage.new("RGB", size, (255, 255, 255))
    side = min(width, height) // 2
    left, top = (width - side) // 2, (height - side) // 2
    ImageDraw.Draw(image).rectangle((left, top, left + side - 1, top + side - 1), fill=color)
    return image


def _user(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


def _try(call: Callable[[], Generation]) -> tuple[Generation | None, str | None]:
    """A generation, or the reason there is none."""
    try:
        return call(), None
    except BackendError as error:
        return None, str(error)


def conforms(text: str, schema: Mapping[str, Any] = SCHEMA) -> bool:
    """Whether ``text`` is a JSON object with exactly the schema's keys, each holding one of its values."""
    try:
        answer = json.loads(text.strip())
    except ValueError:
        return False
    properties = schema["properties"]
    return (
        isinstance(answer, dict)
        and set(answer) == set(properties)
        and all(answer[key] in properties[key]["enum"] for key in properties)
    )


def pixels_per_token(tokens: Mapping[str, int | None]) -> dict[str, Any]:
    """The side of the square of pixels one vision token covers, from the token counts per image size.

    ``proportional`` is false when the counts do not grow with the area, for example because the server resizes
    every image to one size, or tiles it.
    """
    sides = []
    for size, count in tokens.items():
        width, height = (int(part) for part in size.split("x"))
        if count:
            sides.append(math.sqrt(width * height / count))
    if len(sides) < 2:
        return {"side": None, "proportional": None}
    median = statistics.median(sides)
    return {"side": round(median, 1), "proportional": (max(sides) - min(sides)) / median < 0.15}


def run_smoke(backend: Backend, sizes: Sequence[tuple[int, int]] = SIZES) -> dict[str, Any]:
    """Run every check against ``backend`` and return the results. A failing check is recorded, never raised."""
    report: dict[str, Any] = {
        "backend": backend.name,
        "model": backend.model,
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    texts: list[str] = []

    text_only, text_error = _try(lambda: backend.generate(_user(PROBE)))
    report["text_only"] = {"ok": text_only is not None, "error": text_error, "answer": text_only and text_only.text}
    if text_only is not None:
        texts.append(text_only.text)

    with_image, image_error = _try(lambda: backend.generate(_user(PROBE), [square()]))
    report["image"] = {
        "taken": with_image is not None,
        "seen": with_image is not None and "red" in with_image.text.lower(),
        "answer": with_image and with_image.text,
        "error": image_error,
        "seconds": with_image and with_image.seconds,
    }
    if with_image is not None:
        texts.append(with_image.text)

    tokens: dict[str, int | None] = {}
    if with_image is not None and text_only is not None and text_only.prompt_tokens is not None:
        for width, height in sizes:
            image = square((width, height))
            sized, _ = _try(lambda image=image: backend.generate(_user(PROBE), [image], max_tokens=1))
            known = sized is not None and sized.prompt_tokens is not None
            tokens[f"{width}x{height}"] = sized.prompt_tokens - text_only.prompt_tokens if known else None
    report["image_tokens"] = {"per_size": tokens, **pixels_per_token(tokens)}

    report["json_schema"] = check_json_schema(backend, texts)
    report["logprobs"] = check_logprobs(backend)
    report["think_block"] = any(marker in text for text in texts for marker in THINK_MARKERS)
    report["local"] = check_local(backend)
    return report


def check_json_schema(backend: Backend, texts: list[str]) -> dict[str, Any]:
    """Which way of sending a schema the backend enforces. ``local`` enforces none."""
    if not hasattr(backend, "schema_mode"):
        return {"local": "not enforced: transformers has no constrained decoding; parse leniently"}
    results = {}
    original = backend.schema_mode
    try:
        for mode in ("response_format", "guided_json"):
            backend.schema_mode = mode
            answer, error = _try(lambda: backend.generate(_user(SCHEMA_PROBE), [square()], SCHEMA, max_tokens=64))
            if answer is None:
                results[mode] = f"refused: {error}"
                continue
            texts.append(answer.text)
            results[mode] = "enforced" if conforms(answer.text) else f"ignored: answered {answer.text[:60]!r}"
            if results[mode] == "enforced":
                break
    finally:
        backend.schema_mode = original
    return results


def check_logprobs(backend: Backend) -> bool | None:
    """Whether an answer held to the schema carries token logprobs when asked (#75). None: not a chat server."""
    if not hasattr(backend, "extra_body"):
        return None
    original = backend.extra_body
    backend.extra_body = {**original, "logprobs": True}
    try:
        answer, _ = _try(lambda: backend.generate(_user(PROBE), [square()], SCHEMA, max_tokens=64))
    finally:
        backend.extra_body = original
    return bool(answer and answer.logprobs)


def check_local(backend: Backend) -> dict[str, Any] | None:
    """For ``local``: what loaded Apertus (Q-0006), and whether hidden states come out."""
    if backend.name != "local":
        return None
    result: dict[str, Any] = {
        "transformers": getattr(backend, "transformers_version", None),
        "torch": getattr(backend, "torch_version", None),
        "model_class": getattr(getattr(backend, "model_class", None), "__name__", None),
        "device": getattr(backend, "device", None),
        "dtype": getattr(backend, "dtype", None),
        "load_seconds": getattr(backend, "load_seconds", None),
    }
    try:
        features = backend.forward([square()], PROBE)
        result["hidden_states"] = {"shape": list(features.shape)}
    except (BackendError, RuntimeError, ValueError) as error:
        result["hidden_states"] = {"error": f"{type(error).__name__}: {error}"}
    return result


def _seconds(value: float | None) -> str:
    return "–" if value is None else f"{value:.2f} s"


def render_markdown(report: Mapping[str, Any]) -> str:
    """The "For design" block of the B6 PR."""
    image = report["image"]
    if image["seen"]:
        q1 = f"yes, and the model sees it: a red square was answered {image['answer']!r}"
    elif image["taken"]:
        q1 = f"taken, but the answer to a red square was {image['answer']!r}: check that the image reaches the model"
    else:
        q1 = f"no: {image['error']}"
    tokens = report["image_tokens"]
    per_size = ", ".join(f"{size}: {count if count is not None else '?'}" for size, count in tokens["per_size"].items())
    if tokens["side"] and tokens["proportional"]:
        sizing = f"about one token per {tokens['side']:g} x {tokens['side']:g} px, growing with the area"
    elif tokens["side"]:
        sizing = "not proportional to the area: the server resizes or tiles images itself"
    else:
        sizing = "not measured: the backend does not report prompt tokens"
    schema = "; ".join(f"{mode}: {result}" for mode, result in report["json_schema"].items())
    local = report.get("local")
    if local:
        hidden = local["hidden_states"]
        q6 = (
            f"transformers {local['transformers']} (torch {local['torch']}) loaded {report['model']} with "
            f"{local['model_class']} on {local['device']} in {local['dtype']}, {_seconds(local['load_seconds'])}; "
            + (f"hidden states {hidden['shape']}" if "shape" in hidden else f"hidden states failed: {hidden['error']}")
        )
    else:
        q6 = f"not answered by `{report['backend']}`: run `--backend local`"
    think = "yes: turn thinking off in the chat template" if report["think_block"] else "none seen"
    logprobs = {True: "yes", False: "no", None: "not asked: not a chat-completions backend"}[report.get("logprobs")]
    lines = [
        f"**B6 smoke test (#13):** backend `{report['backend']}`, model `{report['model']}`, {report['started']}",
        "",
        f"- **Q-0001 images:** {q1}",
        f"- **Q-0002 JSON schema:** {schema}",
        f"- **Q-0002 tokens per image:** {per_size or 'not measured'}; {sizing}",
        f"- **Token logprobs with the JSON schema (#75):** {logprobs}",
        f"- **Q-0006 transformers:** {q6}",
        f"- **Think block in the answers:** {think}",
        f"- **One request with one image:** {_seconds(image['seconds'])} (latency, not throughput: B7 measures that)",
    ]
    if not report["text_only"]["ok"]:
        lines.insert(2, f"- **The backend did not answer at all:** {report['text_only']['error']}")
    return "\n".join(lines)


def build_backend(args: argparse.Namespace) -> Backend:
    from galaxus.model.apertus import backend

    if args.backend == "api":
        return backend("api")
    if args.backend == "vllm":
        return backend("vllm", base_url=args.base_url, model=args.model or APERTUS_8B)
    return backend(
        "local",
        model=args.model or APERTUS_8B,
        device=args.device,
        dtype=args.dtype,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m galaxus.model.apertus.smoke", description=__doc__.split("\n")[0])
    parser.add_argument("--backend", required=True, choices=("api", "vllm", "local"))
    parser.add_argument("--model", help=f"vllm and local (default {APERTUS_8B}); api takes LLM_NAME")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="vllm: the server's OpenAI-style URL")
    parser.add_argument("--device", help="local: cuda, mps or cpu (default: the best one there is)")
    parser.add_argument("--dtype", default="auto", help="local: auto, bfloat16, float16 or float32")
    parser.add_argument("--revision", help="local: the model revision to load")
    parser.add_argument("--trust-remote-code", action="store_true", help="local: let the model repository run code")
    parser.add_argument("--json", type=Path, help="also write the full report here (outside the repository)")
    args = parser.parse_args(argv)
    try:
        apertus = build_backend(args)
    except BackendError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    report = run_smoke(apertus)
    print(render_markdown(report))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if report["text_only"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
