"""The official metric, loaded from ``$GALAXUS_DATA`` at runtime.

``metric.py`` is Kaggle data: AGENTS.md "Galaxus product rules" 3 forbids copying it into the repository, so it is
imported from the file in ``$GALAXUS_DATA`` instead. It is the same code the leaderboard runs, so a local number
and a leaderboard number differ only in which images they score (see
[challenge_description.md, "Scoring locally"](../../../challenge_description.md#scoring-locally)).

It is loaded under a private module name, so it can never shadow or be shadowed by this module.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

from galaxus.data.classes import ID_COLUMN, dimensions
from galaxus.data.paths import data_file

#: The name ``metric.py`` is registered under in :data:`sys.modules`, chosen not to collide with anything.
METRIC_MODULE_NAME = "galaxus_official_metric"

#: Loaded metrics, keyed by the resolved path of the ``metric.py`` they came from. A single global cached the
#: first file forever, so pointing ``$GALAXUS_DATA`` at another folder kept returning the first folder's metric
#: and its classes (#73).
_loaded: dict[Path, ModuleType] = {}


def load_metric() -> ModuleType:
    """``metric.py`` from ``$GALAXUS_DATA``, loaded once per file and cached by its path.

    Called twice with the same ``$GALAXUS_DATA``, as on Saturday, it reads and executes the file once. Pointed at
    a different ``$GALAXUS_DATA``, it loads that folder's file rather than handing back the first one.

    Raises:
        MissingDataDir: if ``$GALAXUS_DATA`` is unset or unusable.
        FileNotFoundError: if ``metric.py`` is not in ``$GALAXUS_DATA``.
    """
    global _loaded
    if not isinstance(_loaded, dict):
        # Honour the older way of emptying the cache. While this was a single global, callers cleared it by
        # assigning ``None``, and ``tests/serve/conftest.py`` still does to isolate the serve tests from a real
        # ``$GALAXUS_DATA`` (#68, #77). Since #73 the cache is a dict, so that assignment would raise
        # ``AttributeError`` on ``.get`` the moment anything loaded the metric -- and, worse, would silently stop
        # emptying the cache, which is the whole point of that fixture. Anything that is not a dict therefore
        # means "nothing cached". :func:`clear_metric_cache` is the supported way to say it.
        _loaded = {}
    # Resolved before the cache is consulted, because the path is the key: where $GALAXUS_DATA points now is what
    # decides which metric this is, not which one happened to be loaded first.
    path = data_file("metric.py")
    key = Path(path).resolve()
    cached = _loaded.get(key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(METRIC_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} cannot be imported as a Python module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[METRIC_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[METRIC_MODULE_NAME]
        raise
    # The module name is deliberately kept: it shadows nothing. Two different metric.py files in one process
    # therefore both register under it, the later overwriting the earlier in sys.modules, while the module
    # objects themselves stay distinct and cached per path.
    _loaded[key] = module
    return module


def clear_metric_cache() -> None:
    """Forget every ``metric.py`` loaded so far, so the next :func:`load_metric` reads the file again.

    For a test or a notebook that moves ``$GALAXUS_DATA``: the cache is keyed by path, so moving it is enough on
    its own, but clearing is how to force a re-read of the *same* path. Use this rather than assigning to the
    module's private ``_loaded``, which is what the serve tests had to do before this existed (#68).
    """
    global _loaded
    _loaded = {}
    sys.modules.pop(METRIC_MODULE_NAME, None)


def score(solution: pd.DataFrame, submission: pd.DataFrame, id_column: str = ID_COLUMN) -> float:
    """The leaderboard number: the mean of the six macro F1 scores."""
    return load_metric().score(solution, submission, id_column)


def evaluate(solution: pd.DataFrame, submission: pd.DataFrame, id_column: str = ID_COLUMN) -> dict[str, Any]:
    """F1, precision and recall per dimension and per class, as on the Kaggle Results page."""
    return load_metric().evaluate(solution, submission, id_column)


@dataclass(frozen=True)
class DimensionConfusions:
    """What a predictor got wrong in one dimension, on the cells the metric scores (#92).

    ``pairs`` holds ``(label, predicted, count)`` largest first, ties broken by name so the order is stable.
    Correct answers are not pairs: they are counted in :attr:`correct`, so the list is errors only and a
    dimension the predictor never got wrong has an empty one.
    """

    dimension: str
    scored: int
    correct: int
    missing: int
    pairs: tuple[tuple[str, str, int], ...]

    @property
    def wrong(self) -> int:
        """Scored cells answered with the wrong class. Cells with no answer at all are :attr:`missing`."""
        return sum(count for _, _, count in self.pairs)


def _is_blank(value: object) -> bool:
    """True for a cell the predictor left empty: no answer, rather than a wrong answer."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value) == ""


def confusions(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    id_column: str = ID_COLUMN,
    dimension_names: tuple[str, ...] | None = None,
) -> dict[str, DimensionConfusions]:
    """Which class the predictor picked instead of the right one, per dimension, on the scored cells only.

    "Insights: clear evidence of what mattered and what did not" is 15% of the assessment
    ([challenge_description.md, "How we assess"](../../../challenge_description.md#how-we-assess)), and a mean
    macro F1 says a dimension scores badly without saying which classes cause it. This answers that, and tests
    the premise of #48: whether left/right mix-ups are a large share of the perspective errors.

    Only the cells the metric scores are counted. :func:`galaxus.data.split.solution_frame` writes ``None`` into
    the unscored ones, so an unscored cell cannot reach a count here however the predictor answered it.

    A prediction that is not an allowed class is counted exactly as it came, not dropped: a model inventing a
    class name is the kind of thing this table exists to show.

    Args:
        solution: the holdout solution, unscored cells ``None`` (:func:`galaxus.data.split.solution_frame`).
        submission: one row per image, one column per dimension.
        id_column: the column both frames join on.
        dimension_names: which dimensions to count. Defaults to the contract's, in its order.

    Returns:
        One :class:`DimensionConfusions` per dimension present in both frames, keyed by dimension.
    """
    names = tuple(dimension_names) if dimension_names is not None else dimensions()
    merged = solution.merge(submission, on=id_column, suffixes=("_true", "_pred"))
    out: dict[str, DimensionConfusions] = {}
    for dimension in names:
        truth_column, predicted_column = f"{dimension}_true", f"{dimension}_pred"
        if truth_column not in merged.columns or predicted_column not in merged.columns:
            # The dimension is missing from one of the two frames, so there is nothing to compare.
            continue
        scored_rows = merged[merged[truth_column].notna()]
        counts: dict[tuple[str, str], int] = {}
        correct = 0
        missing = 0
        for label, predicted in zip(scored_rows[truth_column], scored_rows[predicted_column], strict=True):
            if _is_blank(predicted):
                missing += 1
            elif str(label) == str(predicted):
                correct += 1
            else:
                key = (str(label), str(predicted))
                counts[key] = counts.get(key, 0) + 1
        pairs = tuple(
            (label, predicted, count)
            for (label, predicted), count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        )
        out[dimension] = DimensionConfusions(dimension, len(scored_rows), correct, missing, pairs)
    return out


def _holdout_solution(dimension_names: tuple[str, ...] | None) -> pd.DataFrame:
    """The holdout solution of the fixed split, with the unscored cells blanked."""
    from galaxus.data.load import read_train
    from galaxus.data.split import solution_frame, split_holdout

    _, holdout = split_holdout(read_train())
    return solution_frame(holdout, dimension_names)


def _wide_predictions(predictions: str, id_column: str = ID_COLUMN) -> pd.DataFrame:
    """An EXP's ``holdout.jsonl`` as one row per image and one column per dimension."""
    from galaxus.data.label_errors import out_of_fold_frame

    # A confusion table needs the class only. The predictors this explains (EXP-0002, EXP-0009) have no
    # confidences until #75 lands, so requiring one here would refuse the very runs #92 is about.
    long = out_of_fold_frame(predictions, require_confidence=False)
    wide = long.pivot_table(index=id_column, columns="dimension", values="prediction", aggfunc="first")
    return wide.reset_index().rename_axis(None, axis="columns")


def main(argv: list[str] | None = None) -> int:
    """``python -m galaxus.data.metric --predictions <EXP folder>``: the confusion pairs per dimension (#92).

    Prints counts and class names only - never an image id (product rule 3) - and writes nothing at all, so it
    cannot put anything into the repository.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m galaxus.data.metric",
        description="Which class a predictor picked instead of the right one, per dimension, on scored cells.",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="holdout.jsonl of an EXP, or the folder holding it ($GALAXUS_DATA/out/predictions/<EXP id>/)",
    )
    parser.add_argument("--top", type=int, default=5, help="pairs to show per dimension (default 5)")
    parser.add_argument("--dimensions", help="comma-separated subset of dimensions (default: all of them)")
    arguments = parser.parse_args(argv)

    if arguments.top < 1:
        parser.error("--top must be at least 1")
    names = tuple(name.strip() for name in arguments.dimensions.split(",")) if arguments.dimensions else None

    try:
        solution = _holdout_solution(names)
        table = confusions(solution, _wide_predictions(arguments.predictions), dimension_names=names)
    except (OSError, KeyError, ValueError, RuntimeError) as failure:
        print(f"cannot work out the confusions: {failure}", file=sys.stderr)
        return 1

    if not table:
        print("[confusions] no dimension is in both the solution and the predictions", file=sys.stderr)
        return 1

    for dimension, found in table.items():
        share = f"{found.correct / found.scored:.1%}" if found.scored else "n/a"
        print(
            f"[confusions] {dimension}: {found.scored} scored, {found.correct} correct ({share}), "
            f"{found.wrong} wrong, {found.missing} without an answer"
        )
        if not found.pairs:
            print("[confusions]   nothing wrong in this dimension")
        for label, predicted, count in found.pairs[: arguments.top]:
            of_wrong = f"{count / found.wrong:.0%}" if found.wrong else "n/a"
            print(f"[confusions]   {label} -> {predicted}: {count} ({of_wrong} of its errors)")
    print("[confusions] counts and class names only; nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
