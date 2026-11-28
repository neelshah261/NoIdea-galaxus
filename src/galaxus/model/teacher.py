"""D (#211): teacher labels on the dev images as extra training rows for the heads, plus self-training.
**Skeleton: the contract only.**

System S (Swarup's desktop) builds this on branch ``model/ad-d-teacher`` (draft PR, #211).

Why it should help: ``gpt-6.2-astra`` with EXP-0049's prompt scores 0.858 on 1,000 random holdout images, where the
70B scores 0.472 (docs/method.md). The labels are learnable from the image, so 5,261 dev images labelled by that
teacher are more, and mostly clean, training rows for every head. The FAQ allows any model in development "to create
training targets or pseudo-labels on the dev images" (FAQ.md:30-31); at runtime only open models classify.

The teacher pass already exists (#172, #173): a person runs it, since it is paid (AGENTS.md rule 7), from a worktree
**without** a ``.env`` and with the teacher's own key (#172: the client rotates every ``LLM_API_KEY*`` it finds):

    LLM_NAME=gpt-6.2-astra LLM_BASE_URL=<OpenAI-compatible URL> LLM_API_KEY=<Parth's teacher key> \\
    ZERO_SHOT_PRESET=zero-shot-observe-first-no-context \\
      uv run python -m galaxus.serve.submit --exp <TEACHER EXP> --part dev --run galaxus.model.zero_shot:from_preset
    # -> $GALAXUS_DATA/out/predictions/<TEACHER EXP>/dev.jsonl  (never upload that submission file)

The contract (fixed by #211 so A and D fit together):

* :func:`teacher_rows`: the teacher's dev answers as rows in train.csv's shape (``image_id``, ``image_file``,
  ``product_id``, the catalog fields, one label per dimension), with ``<dimension>_source = "teacher"`` and
  ``<dimension>_relevant`` from the challenge's scoring rules applied to the teacher's own labels
  (challenge_description.md "Which cells are scored"). The rules come from data (``bench/scoring_rules.json``, #82),
  never typed here. A cell under ``min_confidence`` is left out.
* :func:`training_mask`: the cells a head trains on: human or teacher, and relevant. **Scoring never changes:** EXP
  rows still score only the human cells of the holdout through ``split.scored_mask``.
* :func:`agreeing_rows`: self-training. Dev cells where the members (fused heads and the 70B) agree above
  ``min_confidence``, in the same shape, ``<dimension>_source = "agree"``.
* **Wiring (the only change in helper.py):** ``helper.factory(..., teacher=...)`` appends these rows to the training
  rows (the dev images' embeddings are cached on Neel's laptop); ``HELPER_TEACHER=<dev.jsonl>`` and
  ``HELPER_SELF_TRAIN=1`` turn them on in ``from_env``/``fused_from_env``; eval names ``helper-siglip-teacher`` and
  ``helper-fused-teacher``; whichever of A and D merges second adds ``helper-fused-text-teacher``.
* **With A (#211):** the teacher rows are dev images, which carry the catalog fields, so A's text features reach
  them through ``ContextFeatures.transform``; the text model never trains on teacher labels.
* **Never:** a final image, a holdout image, or a typed class name. The teacher is in MODELS.md as development only.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

#: ``HELPER_TEACHER=<path to the teacher's dev.jsonl>`` adds the teacher's dev rows to the heads' training rows.
TEACHER_ENV = "HELPER_TEACHER"
#: ``HELPER_SELF_TRAIN=1`` adds the dev cells where the members agree.
SELF_TRAIN_ENV = "HELPER_SELF_TRAIN"
TEACHER_SOURCE = "teacher"
AGREE_SOURCE = "agree"

_TODO = "D (#211) is not built yet: system S builds it on branch model/ad-d-teacher"


def teacher_rows(records: Mapping[str, Mapping[str, Any]], dev: pd.DataFrame, min_confidence: float = 0.0) -> pd.DataFrame:
    """The teacher's dev answers as training rows in train.csv's shape (see the module docstring)."""
    raise NotImplementedError(_TODO)


def training_mask(frame: pd.DataFrame, dimension: str) -> pd.Series:
    """The cells a head trains on: a human or teacher label where the dimension is relevant."""
    raise NotImplementedError(_TODO)


def agreeing_rows(members: Mapping[str, Mapping[str, Mapping[str, Any]]], dev: pd.DataFrame,
                  min_confidence: float = 0.9) -> pd.DataFrame:
    """Self-training rows: the dev cells where every member gives the same class above ``min_confidence``."""
    raise NotImplementedError(_TODO)
