"""EXP-0002, zero-shot Apertus v1.5 8B: one call per image, the labeling guideline as the system prompt (#47).

    uv run --env-file .env python -m galaxus.model.eval zero-shot --exp-id EXP-0002
    uv run --env-file .env python -m galaxus.model.eval zero-shot-no-context

- The image is the data lane's prepared JPEG (white compositing, about 512 x 512, through the image cache).
- The system prompt is ``labeling_guideline.md`` read from ``$GALAXUS_DATA`` at runtime, never committed (AGENTS.md
  product rule 3). Its hash goes into the EXP row, so two rows with the same hash saw the same prompt.
- The answer is held to a JSON schema with one enum per dimension, built from ``allowed_classes()``: the gateway
  enforces ``response_format`` (smoke test, #46). Temperature 0. The gateway's ``apertus-v1-5-8b`` is the
  non-thinking model, and the smoke test saw no think block.
- An answer that does not parse, or names a class that does not exist, is asked once more; after that each bad
  dimension gets its default, the majority class of the training rows, and the count goes into the EXP row.
- A backend error (keys spent, server down after its retries) is raised, never turned into defaults: a run that
  silently answers "majority" for half the images would look like a real score.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from galaxus.model.contract import ImageItem, Prediction

GUIDELINE_FILE = "labeling_guideline.md"
IMAGE_SIZE = 512
#: Requests in flight inside one ``predict`` call; lower it if the gateway starts refusing.
WORKERS = 8
#: Gateway retries on 429 / 5xx; with the default 2 a burst of 429s stops the run (#58, Parth on #47).
GATEWAY_RETRIES = 4
MAX_TOKENS = 200
INSTRUCTION = (
    "You label product images for an online shop, following the labeling guideline below exactly.\n"
    "For the image you are given, choose exactly one class for each of these dimensions: {dimensions}.\n"
    "Answer only with a JSON object that has one key per dimension and one of its allowed class names as the value."
)
NO_CONTEXT = "Classify this product image. No product context is available, so judge from the image alone."


def read_guideline() -> str:
    """``labeling_guideline.md`` from ``$GALAXUS_DATA`` (it is Kaggle data, so it is never in the repository)."""
    from galaxus.data.paths import data_file

    return data_file(GUIDELINE_FILE).read_text(encoding="utf-8")


def answer_schema(dimensions: Sequence[str], allowed: Mapping[str, frozenset[str]]) -> dict[str, Any]:
    """A JSON schema with one enum of allowed classes per dimension, every dimension required."""
    return {
        "type": "object",
        "properties": {dimension: {"type": "string", "enum": sorted(allowed[dimension])} for dimension in dimensions},
        "required": list(dimensions),
        "additionalProperties": False,
    }


def system_prompt(guideline: str, dimensions: Sequence[str]) -> str:
    return INSTRUCTION.format(dimensions=", ".join(dimensions)) + "\n\n" + guideline.strip()


def user_prompt(context: Mapping[str, str] | None) -> str:
    """The item's product context as ``name: value`` lines, or the context-free prompt when it has none."""
    if not context:
        return NO_CONTEXT
    lines = "\n".join(f"- {name}: {value}" for name, value in context.items())
    return f"Classify this product image. Product context from the catalog:\n{lines}"


def parse_answer(text: str, dimensions: Sequence[str], allowed: Mapping[str, frozenset[str]]) -> dict[str, str]:
    """The valid classes in ``text``, by dimension. Lenient: code fences and text around the object are ignored.

    A dimension that is missing or names an unknown class is left out, so the caller sees which ones failed.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        answer = json.loads(match.group(0)) if match else None
    except ValueError:
        answer = None
    if not isinstance(answer, dict):
        return {}
    return {
        dimension: str(answer[dimension]).strip()
        for dimension in dimensions
        if str(answer.get(dimension, "")).strip() in allowed[dimension]
    }


class ZeroShotPredictor:
    """Apertus v1.5 classifies each image in one call, with or without the product context.

    Built with no arguments (as the serve runner does: ``--predictor galaxus.model.zero_shot:ZeroShotPredictor``)
    it uses the gateway from ``.env``, the classes and guideline from ``$GALAXUS_DATA`` and the majority answer of
    train minus the holdout as defaults. Thread-safe, so the runner's ``--workers`` may call it in parallel.
    """

    name = "zero-shot"

    def __init__(
        self,
        backend: Any = None,
        *,
        use_context: bool = True,
        defaults: Mapping[str, str] | None = None,
        dimensions: Sequence[str] | None = None,
        allowed: Mapping[str, frozenset[str]] | None = None,
        guideline: str | None = None,
        image_size: int = IMAGE_SIZE,
        workers: int = WORKERS,
    ) -> None:
        if backend is None:
            from galaxus.model.apertus import backend as make_backend

            backend = make_backend("api", retries=GATEWAY_RETRIES)
        if dimensions is None or allowed is None:
            from galaxus.data.classes import allowed_classes
            from galaxus.data.classes import dimensions as read_dimensions

            dimensions = dimensions or read_dimensions()
            allowed = allowed or allowed_classes()
        if defaults is None:
            from galaxus.model.baselines import fit_from_data_lane

            defaults = fit_from_data_lane()
        self.backend = backend
        self.model = backend.model
        self.use_context = use_context
        self.dimensions = tuple(dimensions)
        self.allowed = dict(allowed)
        self.defaults = dict(defaults)
        bad = [d for d in self.dimensions if self.defaults.get(d) not in self.allowed[d]]
        if bad:
            raise ValueError(f"the defaults must be allowed classes, not for {bad}")
        self.image_size = image_size
        self.workers = workers
        self.schema = answer_schema(self.dimensions, self.allowed)
        self.system = system_prompt(guideline if guideline is not None else read_guideline(), self.dimensions)
        self._lock = threading.Lock()
        self.stats = {"images": 0, "retries": 0, "defaulted_cells": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def _image(self, item: ImageItem) -> bytes:
        from galaxus.data.images import cached_jpeg

        return cached_jpeg(item.image_id, item.path, image_size=self.image_size)

    def _ask(self, messages: list[dict[str, str]], image: bytes) -> dict[str, str]:
        generation = self.backend.generate(messages, [image], self.schema, max_tokens=MAX_TOKENS, temperature=0.0)
        with self._lock:
            self.stats["prompt_tokens"] += generation.prompt_tokens or 0
            self.stats["completion_tokens"] += generation.completion_tokens or 0
        return parse_answer(generation.text, self.dimensions, self.allowed)

    def _predict_one(self, item: ImageItem) -> Prediction:
        image = self._image(item)
        context = item.context if self.use_context else None
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": user_prompt(context)}]
        labels = self._ask(messages, image)
        retried = len(labels) < len(self.dimensions)
        if retried:
            labels = {**self._ask(messages, image), **labels}
        missing = [dimension for dimension in self.dimensions if dimension not in labels]
        with self._lock:
            self.stats["images"] += 1
            self.stats["retries"] += retried
            self.stats["defaulted_cells"] += len(missing)
        labels.update({dimension: self.defaults[dimension] for dimension in missing})
        return Prediction(image_id=item.image_id, labels={d: labels[d] for d in self.dimensions})

    def predict(self, batch: Sequence[ImageItem]) -> list[Prediction]:
        """One prediction per item, in order. Up to ``workers`` requests are in flight at once."""
        if self.workers <= 1 or len(batch) <= 1:
            return [self._predict_one(item) for item in batch]
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="zero-shot") as pool:
            return list(pool.map(self._predict_one, batch))

    def prompt_sha256(self) -> str:
        """A short hash of the system prompt and the schema: equal hashes, equal prompts."""
        text = self.system + json.dumps(self.schema, sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def config(self) -> dict[str, Any]:
        """What the EXP row records: model, backend, image size, context, prompt hash, and the answer counts."""
        with self._lock:
            stats = dict(self.stats)
        return {
            "backend": getattr(self.backend, "name", None),
            "image_size": self.image_size,
            "context": self.use_context,
            "temperature": 0.0,
            "max_tokens": MAX_TOKENS,
            "workers": self.workers,
            "prompt_sha256": self.prompt_sha256(),
            **stats,
        }


def no_context() -> ZeroShotPredictor:
    """For the serve runner: ``--predictor galaxus.model.zero_shot:no_context``."""
    return ZeroShotPredictor(use_context=False)
