"""#49: LoRA on Apertus v1.5 8B, the data side and the predictor. No torch here, so ``ci`` tests all of it.

Adapted from reference/finetune_qwen35_reference.ipynb (the hosts' 0.910 recipe), changed where Apertus or our team
rules differ:

- **Images:** always 512 px of area from the data lane's cache (#27), the same pixels the predictor gets.
- **Loss on human labels only:** a class value whose ``<dim>_source`` is ``model`` is masked (label -100) inside the
  answer, so prelabels never teach (the hosts' main lever, #17). The rest of the answer still trains the format.
- **Rare classes repeated** up to 3x towards 200 examples per class; **mirrored rows** for human left/right
  perspectives on text-free images, from the data lane (#29), with the label swapped.
- **The official holdout is left out** of training (data lane split), so EXP-0004 scores it honestly.

Training itself is ``train/lora_apertus.py`` (torch, transformers, peft; GPU). This module also holds the
``LoraPredictor`` that asks the trained model through an OpenAI-compatible endpoint (vLLM serving the merged
weights, ``bench/vllm_apertus.sh MODEL=<merged dir>``) and parses its JSON answer.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

#: Recipe values from the reference notebook (and slide 18 of the deep dive, #17).
RARE_CLASS_TARGET = 200
MAX_REPEATS = 3
IMAGE_SIZE = 512
#: Apertus' vision tokenizer turns 16 x 16 pixels into one token (Q-0002 measures it; this is the plan's number).
PIXELS_PER_TOKEN_SIDE = 16

#: What each dimension asks, in the reference notebook's words (not the labeling guideline, Q-0004).
DIMENSION_MEANINGS = {
    "subject": "what the image shows",
    "background": "the physical backdrop",
    "perspective": "from where the product is seen, judged by where its front points",
    "individuals": "a real person or animal in the photo",
    "text_classification": "text or elements added digitally on top of the photo, or a nutrition label",
    "text_language": "the dominant language of the visible text",
}
CONTEXT_FIELDS = ("product_title", "brand", "product_type", "category", "subcategory")


def system_prompt(classes: Mapping[str, Sequence[str]]) -> str:
    """The instruction, listing the allowed classes read at runtime (never typed here)."""
    lines = [
        f"- {d} ({DIMENSION_MEANINGS.get(d, d)}): {', '.join(sorted(classes[d]))}" for d in classes
    ]
    return (
        "You classify one product image of an online shop along six dimensions. Answer with a JSON object that "
        "holds one class per dimension, in this order:\n" + "\n".join(lines)
    )


def user_prompt(context: Mapping[str, str] | None) -> str:
    """The product context of an image; in production it can be missing."""
    if not context:
        return "Product context: not available."
    parts = [f"{name.replace('_', ' ')}: {context[name]}" for name in CONTEXT_FIELDS if context.get(name)]
    return "\n".join(parts) if parts else "Product context: not available."


def serialize_answer(target: Mapping[str, str]) -> tuple[str, dict[str, tuple[int, int]]]:
    """The answer JSON and, per dimension, the character span of the class value inside its quotes."""
    text, spans = "{", {}
    for index, (dimension, value) in enumerate(target.items()):
        text += ("" if index == 0 else ", ") + json.dumps(dimension) + ": "
        encoded = json.dumps(value)
        spans[dimension] = (len(text) + 1, len(text) + len(encoded) - 1)
        text += encoded
    return text + "}", spans


def parse_answer(answer: str, classes: Mapping[str, Sequence[str]]) -> dict[str, str] | None:
    """The class per dimension of a model answer, or None when it is not a complete, valid answer."""
    try:
        parsed = json.loads(answer[answer.find("{") : answer.rfind("}") + 1])
    except ValueError:
        return None
    if not isinstance(parsed, dict) or any(parsed.get(d) not in set(classes[d]) for d in classes):
        return None
    return {d: parsed[d] for d in classes}


def find_last_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> int:
    """Start of the last occurrence of ``subsequence`` in ``sequence``."""
    n = len(subsequence)
    for start in range(len(sequence) - n, -1, -1):
        if list(sequence[start : start + n]) == list(subsequence):
            return start
    raise ValueError("the answer's tokens are not in the sequence")


def value_positions(
    input_ids: Sequence[int], answer: str, encode_with_offsets: Callable[[str], tuple[list[int], list[tuple[int, int]]]]
) -> tuple[int, dict[str, list[int]]]:
    """(where the answer starts in ``input_ids``, the positions of each dimension's class-value tokens).

    ``encode_with_offsets(text) -> (ids, offsets)`` is the tokenizer without special tokens; the answer is found as the
    last occurrence of its own tokens, as in the reference notebook.
    """
    text, spans = serialize_answer(json.loads(answer))
    ids, offsets = encode_with_offsets(text)
    start = find_last_subsequence(input_ids, ids)
    positions = {
        dimension: [start + i for i, (a, b) in enumerate(offsets) if a < end and b > begin]
        for dimension, (begin, end) in spans.items()
    }
    return start, positions


def training_labels(
    input_ids: Sequence[int],
    answer: str,
    trained_dimensions: Sequence[str],
    encode_with_offsets: Callable[[str], tuple[list[int], list[tuple[int, int]]]],
) -> list[int]:
    """Labels for one example: -100 before the answer (no loss on the prompt or the image) and on the class values
    of prelabelled dimensions; the token ids everywhere else in the answer."""
    start, positions = value_positions(input_ids, answer, encode_with_offsets)
    labels = [-100] * start + list(input_ids[start:])
    for dimension, places in positions.items():
        if dimension not in trained_dimensions:
            for place in places:
                labels[place] = -100
    return labels


@dataclass(frozen=True)
class Example:
    """One training example: which image (mirrored or not), its context, the answer, the dimensions in the loss."""

    image_id: str
    image_file: str
    mirrored: bool
    context: dict[str, str] | None
    target: dict[str, str]
    trained_dimensions: tuple[str, ...]

    @property
    def answer(self) -> str:
        return serialize_answer(self.target)[0]


@dataclass
class TrainingSet:
    examples: list[Example]
    stats: dict[str, Any] = field(default_factory=dict)


def _context(row: Mapping[str, Any]) -> dict[str, str] | None:
    context = {name: str(row[name]) for name in CONTEXT_FIELDS if name in row and str(row[name]).strip()}
    return context or None


def build_training_set(
    rows: pd.DataFrame,
    dimensions: Sequence[str],
    mirrored: pd.DataFrame | None = None,
    rare_target: int = RARE_CLASS_TARGET,
    max_repeats: int = MAX_REPEATS,
) -> TrainingSet:
    """Examples for one epoch: every row once, rows of rare human classes repeated, plus the mirrored rows.

    ``rows`` is train minus the official holdout; ``mirrored`` is the data lane's ``mirrored_rows(rows)``.
    """
    records = rows.to_dict(orient="records")
    human = [tuple(d for d in dimensions if r.get(f"{d}_source") == "human") for r in records]
    counts = Counter((d, r[d]) for r, dims in zip(records, human) for d in dims)

    def repeats(record: Mapping[str, Any], dims: Sequence[str]) -> int:
        rarest = min((counts[(d, record[d])] for d in dims), default=rare_target)
        return min(max_repeats, max(1, math.ceil(rare_target / max(1, rarest))))

    examples: list[Example] = []
    for record, dims in zip(records, human):
        example = Example(
            image_id=str(record["image_id"]),
            image_file=str(record.get("image_file", "")),
            mirrored=False,
            context=_context(record),
            target={d: str(record[d]) for d in dimensions},
            trained_dimensions=dims,
        )
        examples.extend([example] * repeats(record, dims))
    added_mirrors = 0
    if mirrored is not None and len(mirrored):
        for record in mirrored.to_dict(orient="records"):
            dims = tuple(d for d in dimensions if record.get(f"{d}_source") == "human")
            examples.append(
                Example(str(record["image_id"]), str(record.get("image_file", "")), True, _context(record),
                        {d: str(record[d]) for d in dimensions}, dims)
            )
            added_mirrors += 1
    stats = {
        "images": len(records),
        "repeats_added": len(examples) - len(records) - added_mirrors,
        "mirrored_added": added_mirrors,
        "examples_per_epoch": len(examples),
        "human_labels_in_loss": {d: sum(d in dims for dims in human) for d in dimensions},
        "prelabels_masked": {d: sum(d not in dims for dims in human) for d in dimensions},
    }
    return TrainingSet(examples, stats)


# ---------------------------------------------------------------- #135: class weights and focal loss (#17, idea 5)

#: (n / (k * count)) ** power, the heads' tempered class weights (#51): 0 = none, 1 = fully balanced, 0.5 tempers it.
CLASS_WEIGHT_POWER = 0.5
CLASS_WEIGHTINGS = ("none", "tempered")


def class_weights(examples: Sequence[Example], power: float = CLASS_WEIGHT_POWER) -> dict[str, dict[str, float]]:
    """Per dimension, a loss weight per class: ``(n / (k * count)) ** power``, scaled so the weight averaged over the
    trained cells is 1 (the loss keeps its size; rare classes weigh more, common ones less). The same formula as the
    heads (#51).

    Counted on the epoch as built: train minus the official holdout, with the rare-class repeats and the mirrored rows,
    on the trained (human) dimensions only. So the weights correct what the loss actually sees, and the repeats are not
    counted twice.
    """
    counts: dict[str, Counter] = {}
    for example in examples:
        for dimension in example.trained_dimensions:
            counts.setdefault(dimension, Counter())[example.target[dimension]] += 1
    weights: dict[str, dict[str, float]] = {}
    for dimension, counter in counts.items():
        n, k = sum(counter.values()), len(counter)
        raw = {name: (n / (k * count)) ** power for name, count in counter.items()}
        mean = sum(raw[name] * count for name, count in counter.items()) / n
        weights[dimension] = {name: value / mean for name, value in raw.items()}
    return weights


@dataclass(frozen=True)
class LossOptions:
    """What the LoRA loss adds to plain cross-entropy (#135). The default is the plain loss, the current recipe."""

    class_weights: dict[str, dict[str, float]] | None = None
    focal_gamma: float = 0.0

    @property
    def plain(self) -> bool:
        return self.class_weights is None and self.focal_gamma == 0

    def describe(self) -> str:
        if self.plain:
            return "plain cross-entropy"
        weighting = f"tempered class weights (power {CLASS_WEIGHT_POWER})" if self.class_weights else "no class weights"
        return f"{weighting}, focal gamma {self.focal_gamma:g} on the class-value tokens"


def loss_options(examples: Sequence[Example], class_weighting: str = "none", focal_gamma: float = 0.0,
                 power: float = CLASS_WEIGHT_POWER) -> LossOptions:
    """The ``--class-weighting`` and ``--focal-gamma`` flags of ``train/lora_apertus.py``, checked."""
    if class_weighting not in CLASS_WEIGHTINGS:
        raise ValueError(f"class weighting {class_weighting!r} is not one of {', '.join(CLASS_WEIGHTINGS)}")
    if focal_gamma < 0:
        raise ValueError("the focal gamma is >= 0 (0 = off)")
    weights = class_weights(examples, power) if class_weighting == "tempered" else None
    return LossOptions(weights, float(focal_gamma))


def loss_weights(
    input_ids: Sequence[int],
    example: Example,
    encode_with_offsets: Callable[[str], tuple[list[int], list[tuple[int, int]]]],
    weights: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, list]:
    """Per position of ``input_ids``: ``loss_weights``, the class weight on each class-value token of a trained
    dimension and 1 everywhere else, and ``value_mask``, 1 on those same tokens (where focal loss applies).

    Prompt, image and prelabel positions keep weight 1, but their labels are -100 (``training_labels``), so they carry no
    loss: the masking is unchanged.
    """
    _, positions = value_positions(input_ids, example.answer, encode_with_offsets)
    weight, mask = [1.0] * len(input_ids), [0] * len(input_ids)
    for dimension in example.trained_dimensions:
        value = weights.get(dimension, {}).get(example.target[dimension], 1.0) if weights else 1.0
        for place in positions.get(dimension, ()):
            weight[place], mask[place] = value, 1
    return {"loss_weights": weight, "value_mask": mask}


def combine_token_losses(token_loss: Any, valid: Any, weight: Any, value_mask: Any, focal_gamma: float, xp: Any):
    """(sum of the weighted per-token losses, number of tokens in the loss) for one batch.

    ``token_loss`` is the plain cross-entropy per position (``-log p`` of the label, 0 where the label is -100) and
    ``valid`` marks the positions in the loss. When ``focal_gamma`` > 0, focal loss ``(1 - p) ** gamma * -log p`` replaces
    it on the class-value tokens. Gamma 0 with weights of 1 gives back the plain loss. ``xp`` is ``numpy`` or ``torch``:
    the same arithmetic runs on both, so ci tests it without torch.
    """
    if focal_gamma > 0:
        focal = (1 - xp.exp(-token_loss)) ** focal_gamma * token_loss
        token_loss = xp.where(value_mask > 0, focal, token_loss)
    return (token_loss * weight * valid).sum(), valid.sum()


# ---------------------------------------------------------------- the launch plan and its cost


@dataclass(frozen=True)
class Plan:
    examples_per_epoch: int
    epochs: int
    batch: int
    accumulation: int
    tokens_per_example: int
    tokens_per_second: float
    dollars_per_hour: float

    @property
    def steps(self) -> int:
        return math.ceil(self.examples_per_epoch / (self.batch * self.accumulation)) * self.epochs

    @property
    def hours(self) -> float:
        return self.examples_per_epoch * self.epochs * self.tokens_per_example / self.tokens_per_second / 3600

    @property
    def dollars(self) -> float:
        return self.hours * self.dollars_per_hour


def tokens_per_example(image_size: int = IMAGE_SIZE, prompt_tokens: int = 450, answer_tokens: int = 70) -> int:
    """Image tokens at ``image_size`` px of area plus the prompt and the answer (estimates until Q-0002 measures)."""
    return (image_size // PIXELS_PER_TOKEN_SIDE) ** 2 + prompt_tokens + answer_tokens


# ---------------------------------------------------------------- predictor for the trained model


class LoraPredictor:
    """The contract over an OpenAI-compatible endpoint serving the merged LoRA model (vLLM).

    One request per image with the training prompt. The request carries a JSON schema with one enum of allowed classes
    per dimension (#100), so vLLM can only decode valid answers. The answer is still parsed and checked: an invalid one
    (a backend that ignores the schema, a cut-off answer) falls back to ``fallback`` (the majority answer), so a cell
    is never empty, and the count of fallbacks is kept for the report.
    """

    name = "lora"

    def __init__(self, backend: Any, classes: Mapping[str, Sequence[str]], fallback: Mapping[str, str],
                 image_bytes: Callable[[Any], bytes], model: str | None = None) -> None:
        from galaxus.model.zero_shot import answer_schema

        self.backend = backend
        self.classes = {d: list(v) for d, v in classes.items()}
        self.fallback = dict(fallback)
        self.image_bytes = image_bytes
        self.model = model
        self.system = system_prompt(self.classes)
        self.schema = answer_schema(list(self.classes), {d: frozenset(v) for d, v in self.classes.items()})
        self.fallbacks = 0
        self.no_confidence_cells = 0

    def predict(self, batch: Sequence[Any]) -> list[Any]:
        """One ``galaxus.types.Prediction`` per item, in order (the contract ``eval``, ``runner`` and ``submit`` read)."""
        from galaxus.model.zero_shot import value_confidence
        from galaxus.types import Prediction

        answers = []
        for item in batch:
            generation = self.backend.generate(
                [{"role": "system", "content": self.system},
                 {"role": "user", "content": user_prompt(getattr(item, "context", None))}],
                [self.image_bytes(item)],
                json_schema=self.schema,
                max_tokens=160,
                temperature=0.0,
            )
            labels = parse_answer(generation.text, self.classes)
            confidence: dict[str, float] = {}
            if labels is None:
                self.fallbacks += 1
                labels = dict(self.fallback)
            else:
                # #75: P(class | answer so far) from the answer's token logprobs, as the zero-shot predictor does.
                # None when the backend sent none; a fallback answer never gets a made-up number.
                confidence = value_confidence(getattr(generation, "logprobs", None), labels)
            self.no_confidence_cells += len(labels) - len(confidence)
            answers.append(Prediction(image_id=item.image_id, labels=labels, confidence=confidence or None))
        return answers

    def config(self) -> dict[str, object]:
        return {"fallbacks": self.fallbacks, "no_confidence_cells": self.no_confidence_cells, "image_size": IMAGE_SIZE,
                "json_schema": "enum per dimension", "confidence": "token logprobs of the class name (#75)"}


# ---------------------------------------------------------------- EXP-0004: the eval script's "lora" predictor

#: Where vLLM serves the merged LoRA weights: ``MODEL=<run>/merged bash bench/vllm_apertus.sh`` on the GPU machine.
BASE_URL_ENV = "LORA_BASE_URL"
SERVED_NAME_ENV = "LORA_SERVED_MODEL"
DEFAULT_SERVED_NAME = "apertus-lora"
#: A bearer token for an endpoint behind an authenticating proxy, e.g. an HF Job's exposed port (#76), which takes
#: an HF token with read access to the job's namespace. Read from the environment, never logged.
API_KEY_ENV = "LORA_API_KEY"
#: Same as the Saturday run sheet (#58): a 429 burst longer than 2 retries stops a run.
RETRIES = 4


def eval_factory(environ: Mapping[str, str] | None = None, backend: Any = None) -> Callable[..., LoraPredictor]:
    """``factory(training_rows, data, seed)`` for ``galaxus.model.eval.PREDICTORS["lora"]``.

    Reads the endpoint from ``$LORA_BASE_URL`` (e.g. ``http://<gpu-host>:8000/v1``); ``backend`` replaces it in tests.
    The fallback answer is the majority answer of the training rows, as for the zero-shot predictor.
    """
    import os

    settings = dict(os.environ if environ is None else environ)

    def factory(rows: pd.DataFrame, data: Any, seed: int) -> LoraPredictor:
        from galaxus.model.baselines import MajorityPredictor

        chosen = backend if backend is not None else _vllm_backend(settings)
        dimensions = list(data.dimensions())
        allowed = data.allowed_classes()
        fallback = MajorityPredictor.fit(rows, dimensions, data.scored_mask).answer
        return LoraPredictor(chosen, {d: sorted(allowed[d]) for d in dimensions}, fallback, _cached_image,
                             model=getattr(chosen, "model", None))

    return factory


def _vllm_backend(settings: Mapping[str, str]) -> Any:
    """The vLLM endpoint from ``$LORA_BASE_URL`` (and ``$LORA_API_KEY`` when it sits behind a proxy that wants one).

    Checked first, so a missing endpoint is the first, clearest error.
    """
    base_url = settings.get(BASE_URL_ENV)
    if not base_url:
        raise RuntimeError(
            f"set ${BASE_URL_ENV} to the vLLM endpoint serving the merged LoRA weights, e.g. http://<gpu-host>:8000/v1 "
            f"(started with MODEL=<run>/merged bash bench/vllm_apertus.sh)"
        )
    from galaxus.model.apertus.vllm import VllmBackend

    return VllmBackend(base_url, settings.get(SERVED_NAME_ENV, DEFAULT_SERVED_NAME),
                       api_key=settings.get(API_KEY_ENV) or None, retries=RETRIES,
                       extra_body={"logprobs": True})  # #75: confidences, for mirror TTA (#48) and calibration (#42)


def from_env(environ: Mapping[str, str] | None = None, backend: Any = None) -> LoraPredictor:
    """The LoRA predictor with no arguments, for ``submit --run`` and ``final_run --predictor`` (#100).

        python -m galaxus.serve.submit --exp EXP-0004 --part dev --run galaxus.model.lora:from_env
        python -m galaxus.serve.final_run --drill --predictor galaxus.model.lora:from_env --exp EXP-0004

    The endpoint comes from ``$LORA_BASE_URL`` and ``$LORA_SERVED_MODEL`` (checked before anything else is read), the
    classes from the contract (``galaxus.data.classes``), the fallback from ``MajorityPredictor()`` (train minus the
    official holdout), and the images from the same 512 px cache as ``eval_factory``. So EXP-0004's holdout score,
    the dev file and Saturday's final file come from identical prompts and identical pixels.
    """
    import os

    from galaxus.data.classes import allowed_classes, dimensions
    from galaxus.model.baselines import MajorityPredictor

    settings = dict(os.environ if environ is None else environ)
    chosen = backend if backend is not None else _vllm_backend(settings)
    names = list(dimensions())
    allowed = allowed_classes()
    fallback = MajorityPredictor().answer
    return LoraPredictor(chosen, {d: sorted(allowed[d]) for d in names}, fallback, _cached_image,
                         model=getattr(chosen, "model", None))


def _cached_image(item: Any) -> bytes:
    from galaxus.data.images import cached_jpeg

    return cached_jpeg(item.image_id, item.path, image_size=IMAGE_SIZE)
