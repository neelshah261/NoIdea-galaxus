"""Synthetic helpers for the serve-lane tests (fixtures are in conftest.py). No competition data, no network (AGENTS.md "Work → PR" 5)."""

from __future__ import annotations

from dataclasses import dataclass

DIMENSIONS = ("subject", "background", "perspective", "individuals", "text_classification", "text_language")
# Invented class names: only self-consistent. Real names are read at runtime, never written down (AGENTS.md).
FAKE_CLASSES = {dimension: [f"{dimension}_a", f"{dimension}_b", "none"] for dimension in DIMENSIONS}
DEV_IDS = ["img_d1", "img_d2", "img_d3"]
FINAL_IDS = ["img_f1", "img_f2"]


@dataclass(frozen=True)
class Item:
    image_id: str
    path: str
    context: dict[str, str] | None = None


@dataclass(frozen=True)
class Pred:
    image_id: str
    labels: dict[str, str]
    confidence: dict[str, float] | None = None


class StubPredictor:
    """Answers the second fake class in every dimension and counts what it was asked."""

    def __init__(self, crash_after_batches: int | None = None) -> None:
        self.calls: list[list[str]] = []
        self.crash_after_batches = crash_after_batches

    def predict(self, items):
        if self.crash_after_batches is not None and len(self.calls) >= self.crash_after_batches:
            raise KeyboardInterrupt("simulated kill")
        self.calls.append([item.image_id for item in items])
        return [Pred(item.image_id, {d: FAKE_CLASSES[d][1] for d in DIMENSIONS}, {d: 0.9 for d in DIMENSIONS})
                for item in items]


def stub_factory() -> StubPredictor:
    """Factory used by load_predictor tests: ``serve_helpers:stub_factory``."""
    return StubPredictor()
