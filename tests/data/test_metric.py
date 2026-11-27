"""Issue #5: the metric wrapper loads metric.py from $GALAXUS_DATA and never from the repository.

The real metric.py is Kaggle data, so these tests use a stub with the same two entry points. The test that needs
the real file is skipped when ``$GALAXUS_DATA`` has no metric.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from galaxus.data import metric as metric_module
from galaxus.data.metric import METRIC_MODULE_NAME, evaluate, load_metric, score
from galaxus.data.paths import GALAXUS_DATA_ENV, MissingDataDir

STUB = """
DIMENSIONS = ("subject", "background")
VALID_CLASSES = {"subject": ["alpha", "beta"], "background": ["white", "color"]}


def score(solution, submission, id_column):
    return 0.5


def evaluate(solution, submission, id_column):
    return {"overall": {"f1_macro": 0.5}}
"""


@pytest.fixture(autouse=True)
def fresh_cache():
    """The module caches the loaded metric; every test starts without it."""
    metric_module._loaded = None
    sys.modules.pop(METRIC_MODULE_NAME, None)
    yield
    metric_module._loaded = None
    sys.modules.pop(METRIC_MODULE_NAME, None)


def test_metric_py_is_not_in_the_repository():
    """AGENTS.md "Galaxus product rules" 3: metric.py is loaded from $GALAXUS_DATA, never committed."""
    repository = Path(__file__).resolve().parents[2]
    assert not (repository / "metric.py").exists()


def test_unset_data_dir_is_a_clear_error(monkeypatch):
    monkeypatch.delenv(GALAXUS_DATA_ENV, raising=False)
    with pytest.raises(MissingDataDir):
        load_metric()


def test_missing_metric_py_is_a_clear_error(galaxus_data):
    with pytest.raises(FileNotFoundError, match="metric.py"):
        load_metric()


def test_the_stub_is_loaded_from_the_data_dir(galaxus_data):
    (galaxus_data / "metric.py").write_text(STUB, encoding="utf-8")
    loaded = load_metric()
    assert loaded.DIMENSIONS == ("subject", "background")
    assert Path(loaded.__file__) == galaxus_data / "metric.py"


def test_it_is_loaded_under_a_name_that_cannot_shadow_our_module(galaxus_data):
    (galaxus_data / "metric.py").write_text(STUB, encoding="utf-8")
    load_metric()
    assert METRIC_MODULE_NAME in sys.modules
    assert sys.modules["galaxus.data.metric"] is metric_module


def test_score_and_evaluate_are_forwarded(galaxus_data):
    (galaxus_data / "metric.py").write_text(STUB, encoding="utf-8")
    frame = pd.DataFrame({"image_id": ["img_1"], "subject": ["alpha"]})
    assert score(frame, frame) == 0.5
    assert evaluate(frame, frame) == {"overall": {"f1_macro": 0.5}}


def test_a_broken_metric_py_is_not_left_half_imported(galaxus_data):
    (galaxus_data / "metric.py").write_text("raise ValueError('broken')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="broken"):
        load_metric()
    assert METRIC_MODULE_NAME not in sys.modules


@pytest.mark.skipif(
    not (os.environ.get(GALAXUS_DATA_ENV) and Path(os.environ[GALAXUS_DATA_ENV], "metric.py").is_file()),
    reason=f"the real metric.py is only available where ${GALAXUS_DATA_ENV} holds the Kaggle files",
)
def test_the_real_metric_exposes_the_entry_points_we_use():
    loaded = load_metric()
    assert len(loaded.DIMENSIONS) == 6
    assert callable(loaded.score) and callable(loaded.evaluate)


# --- Parity with the real metric.py -------------------------------------------------------------------------
#
# These are the checks that decide whether our local score equals the leaderboard's. They need the Kaggle files,
# so they skip everywhere except a machine with $GALAXUS_DATA (issue #68). They are named "parity" so that
# ``pytest tests/data -k parity`` selects them.

_HAS_REAL_METRIC = (
    bool(os.environ.get(GALAXUS_DATA_ENV)) and Path(os.environ.get(GALAXUS_DATA_ENV, ""), "metric.py").is_file()
)
_needs_real_data = pytest.mark.skipif(
    not _HAS_REAL_METRIC,
    reason=f"parity needs the Kaggle files: set ${GALAXUS_DATA_ENV} to a directory holding metric.py and train.csv",
)


def _real_holdout_and_submission():
    """The official holdout, our solution frame for it, and a constant-answer submission over the same ids."""
    from galaxus.data.classes import allowed_classes, dimensions
    from galaxus.data.load import read_train
    from galaxus.data.split import solution_frame, split_holdout

    _, holdout = split_holdout(read_train())
    names = dimensions()
    solution = solution_frame(holdout, names)
    allowed = allowed_classes()
    answer = {name: min(allowed[name]) for name in names}
    submission = pd.DataFrame({"image_id": solution["image_id"], **{n: answer[n] for n in names}})
    return holdout, solution, submission, names, allowed


@_needs_real_data
def test_parity_our_wrapper_returns_exactly_what_metric_py_returns():
    """Guards against the wrapper quietly growing post-processing of its own."""
    _, solution, submission, _, _ = _real_holdout_and_submission()
    assert score(solution, submission) == load_metric().score(solution, submission, "image_id")


@_needs_real_data
def test_parity_an_unscored_cell_cannot_change_the_score():
    """The assumption our whole local score rests on.

    ``solution_frame`` puts ``None`` in every cell the labeling rules do not score. If Kaggle's ``metric.py``
    counted those cells instead of skipping them, our local number would drift from the leaderboard and we would
    not find out until a submission came back. So: change the prediction in *only* the unscored cells and the
    score must not move.
    """
    _, solution, submission, names, allowed = _real_holdout_and_submission()
    before = score(solution, submission)

    changed = submission.copy()
    touched = 0
    for name in names:
        other = max(allowed[name])
        unscored = solution[name].isna()
        touched += int(unscored.sum())
        changed.loc[unscored.to_numpy(), name] = other

    assert touched > 0, "no unscored cell in the holdout, so this check proves nothing - investigate"
    assert score(solution, changed) == before


@_needs_real_data
def test_parity_a_scored_cell_does_change_the_score():
    """The other half: if nothing moves the score, the test above would pass for the wrong reason."""
    _, solution, submission, names, allowed = _real_holdout_and_submission()
    before = score(solution, submission)

    changed = submission.copy()
    for name in names:
        other = max(allowed[name])
        scored = solution[name].notna()
        changed.loc[scored.to_numpy(), name] = other

    assert score(solution, changed) != before


@_needs_real_data
def test_parity_evaluate_covers_every_dimension():
    """`evaluate` feeds the per-dimension rows of the report and the deck, so it must cover all six."""
    from galaxus.data.classes import dimensions

    _, solution, submission, _, _ = _real_holdout_and_submission()
    report = evaluate(solution, submission)
    flat = str(report)
    for name in dimensions():
        assert name in flat, f"{name} missing from evaluate()'s report"


@_needs_real_data
def test_parity_unscored_cells_are_skipped_dimension_by_dimension():
    """The sharp version, and the one that decides whether our local score matches the leaderboard.

    Changing predictions *inside* unscored cells can, where the class distribution is balanced, move the penalty
    from one class to another and leave the mean untouched - so that check alone can pass against a ``metric.py``
    that wrongly counts unscored cells. This one scores a single dimension twice: once over every row, once with
    that dimension's unscored rows removed outright. If nulls are skipped the two are identical; if they are
    counted, they differ.
    """
    _, solution, submission, names, _ = _real_holdout_and_submission()
    checked = 0
    for name in names:
        unscored = solution[name].isna()
        if not unscored.any() or unscored.all():
            continue
        checked += 1
        one = solution[["image_id", name]]
        with_nulls = score(one, submission)
        kept = one[~unscored.to_numpy()].reset_index(drop=True)
        without = score(kept, submission[submission["image_id"].isin(kept["image_id"])].reset_index(drop=True))
        assert without == with_nulls, (
            f"{name}: scoring it over every row gives {with_nulls}, but dropping its unscored rows gives "
            f"{without}. metric.py is counting cells our solution leaves empty, so our local score and the "
            "leaderboard's will not agree"
        )
    assert checked > 0, "no dimension had a mix of scored and unscored cells, so this proves nothing"
