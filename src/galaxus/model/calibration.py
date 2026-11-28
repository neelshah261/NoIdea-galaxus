"""#42: per-class bias on log-probabilities, fitted for macro F1 (plan #40, option 2).

Macro F1 weighs every class of a dimension equally, so a model that is well calibrated for accuracy under-predicts
the rare classes. Adding a bias ``b[c]`` to ``log p(c)`` before the argmax moves the decision boundaries; the biases
are fitted by coordinate ascent on macro F1. Fit them on **out-of-fold** predictions only (grouped folds of train
minus the holdout), never on dev or final images: Q-0005 in PLAN.md is still open with the hosts.

Pure numpy, runs on CPU in seconds.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: Bias values tried for each class, in log-probability units.
GRID = np.round(np.arange(-3.0, 3.01, 0.25), 2)
ROUNDS = 3


def macro_f1(true: np.ndarray, predicted: np.ndarray) -> float:
    """Macro F1 over every class that is true or predicted, as metric.py counts it."""
    classes = np.union1d(true, predicted)
    scores = []
    for value in classes:
        tp = np.sum((predicted == value) & (true == value))
        fp = np.sum((predicted == value) & (true != value))
        fn = np.sum((predicted != value) & (true == value))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores)) if scores else 0.0


def apply_bias(log_probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Class index per row after the bias."""
    return np.argmax(log_probs + bias[None, :], axis=1)


def fit_bias(log_probs: np.ndarray, true: np.ndarray, grid: Sequence[float] = GRID, rounds: int = ROUNDS) -> np.ndarray:
    """Biases (one per class) that maximise macro F1 of ``argmax(log_probs + bias)`` against ``true``.

    Coordinate ascent: each round tries every grid value for each class in turn and keeps the best. Ties keep the
    current value, so a bias only moves when macro F1 strictly improves, and an empty input returns zeros.
    """
    classes = log_probs.shape[1]
    bias = np.zeros(classes, dtype=np.float64)
    if len(true) == 0:
        return bias
    best = macro_f1(true, apply_bias(log_probs, bias))
    for _ in range(rounds):
        improved = False
        for index in range(classes):
            current = bias[index]
            for value in grid:
                if value == current:
                    continue
                bias[index] = value
                score = macro_f1(true, apply_bias(log_probs, bias))
                if score > best + 1e-12:
                    best, current, improved = score, value, True
            bias[index] = current
        if not improved:
            break
    return bias
