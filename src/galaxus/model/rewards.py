"""#134: RL with verifiable rewards (the hosts' idea 4, #17). The reward and the data side of GRPO; no torch here.

GRPO samples several answers per image from the LoRA and pushes the model towards the better ones in each group. The
reward is checked against the human labels, so it can't be gamed:

* **Invalid answer** (not JSON, a missing dimension, a class that doesn't exist): 0.
* **Valid answer:** a small format bonus, plus, for each human-labelled (trained) dimension it gets right, that true
  class's weight from #135's tempered class weights. A rare class pays more than a common one, which is what macro F1
  counts. Prelabelled cells pay nothing, as in the LoRA's loss.

GRPO normalises rewards within each group of answers to the same image, so only the differences between answers count:
for an image whose true class is rare, getting that class right is what moves the model.

Training itself is ``train/grpo_apertus.py`` (TRL's ``GRPOTrainer``, GPU). The prompt, pixels and answer format are
the LoRA's (``galaxus.model.lora``), so the result is served and scored exactly like EXP-0004.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from galaxus.model.lora import Example, parse_answer

#: What a valid but entirely wrong answer earns, so the model keeps the format while it explores.
FORMAT_BONUS = 0.1
#: Answers sampled per image. More gives GRPO a better baseline but costs generation time.
GENERATIONS = 4
#: The training run (#76 run 3) measured 1.13 examples/s on one L40S; GRPO generates ``GENERATIONS`` answers and
#: trains on all of them, so it is taken as that many times slower, with 1.5x for the generation itself (an estimate:
#: the run prints the measured rate).
SFT_EXAMPLES_PER_SECOND_L40S = 1.13
GENERATION_OVERHEAD = 1.5


def _text(completion: Any) -> str:
    """A completion as text: TRL passes a string, or a list of chat messages for conversational datasets."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Sequence) and completion and isinstance(completion[-1], Mapping):
        return str(completion[-1].get("content", ""))
    return str(completion)


def answer_reward(completion: Any, target: Mapping[str, str], trained: Sequence[str],
                  classes: Mapping[str, Sequence[str]], weights: Mapping[str, Mapping[str, float]] | None = None,
                  format_bonus: float = FORMAT_BONUS) -> float:
    """The reward for one answer to one image (see the module docstring)."""
    labels = parse_answer(_text(completion), classes)
    if labels is None:
        return 0.0
    if not trained:
        return format_bonus
    earned = sum((weights or {}).get(d, {}).get(target[d], 1.0) for d in trained if labels.get(d) == target[d])
    return format_bonus + earned / len(trained)


def reward_function(classes: Mapping[str, Sequence[str]],
                    weights: Mapping[str, Mapping[str, float]] | None = None) -> Callable[..., list[float]]:
    """A reward function in TRL's shape: ``f(prompts, completions, target, trained, **columns) -> rewards``.

    ``target`` and ``trained`` are dataset columns (one per prompt, repeated for each of its completions by TRL), as
    :func:`grpo_rows` builds them.
    """

    def reward(prompts: Sequence[Any], completions: Sequence[Any], target: Sequence[Mapping[str, str]],
               trained: Sequence[Sequence[str]], **_: Any) -> list[float]:
        return [answer_reward(c, t, d, classes, weights) for c, t, d in zip(completions, target, trained, strict=True)]

    reward.__name__ = "human_label_reward"
    return reward


def pick_examples(examples: Sequence[Example], count: int, weights: Mapping[str, Mapping[str, float]] | None,
                  seed: int = 0) -> list[Example]:
    """``count`` distinct images for GRPO, drawn without replacement with probability growing with the rarest trained
    class's weight (Efraimidis-Spirakis keys). Rare classes are where macro F1 is won; common ones still appear."""
    rng = random.Random(seed)
    unique = {e.image_id: e for e in examples if not e.mirrored and e.trained_dimensions}
    keyed = []
    for example in unique.values():
        weight = max((weights or {}).get(d, {}).get(example.target[d], 1.0) for d in example.trained_dimensions)
        keyed.append((rng.random() ** (1.0 / max(weight, 1e-6)), example.image_id))
    keyed.sort(reverse=True)
    return [unique[image_id] for _, image_id in keyed[:count]]


def grpo_rows(examples: Sequence[Example]) -> list[dict[str, Any]]:
    """One dataset row per image: what the prompt needs and what the reward checks."""
    return [{"image_id": e.image_id, "image_file": e.image_file, "context": e.context, "target": dict(e.target),
             "trained": list(e.trained_dimensions)} for e in examples]


@dataclass(frozen=True)
class GrpoPlan:
    images: int
    generations: int
    dollars_per_hour: float

    @property
    def images_per_second(self) -> float:
        return SFT_EXAMPLES_PER_SECOND_L40S / (self.generations * GENERATION_OVERHEAD)

    @property
    def hours(self) -> float:
        return self.images / self.images_per_second / 3600

    @property
    def dollars(self) -> float:
        return self.hours * self.dollars_per_hour


def images_for_budget(dollars: float, generations: int = GENERATIONS, dollars_per_hour: float = 1.8) -> int:
    """How many images one GRPO pass can take for ``dollars`` at the estimated rate."""
    per_image = GrpoPlan(1, generations, dollars_per_hour).dollars
    count = max(0, math.floor(dollars / per_image))
    while count > 0 and GrpoPlan(count, generations, dollars_per_hour).dollars > dollars:  # floating-point rounding
        count -= 1
    return count
