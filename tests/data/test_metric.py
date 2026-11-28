"""Issue #5: the metric wrapper loads metric.py from $GALAXUS_DATA and never from the repository.

The real metric.py is Kaggle data, so these tests use a stub with the same two entry points. The test that needs
the real file is skipped when ``$GALAXUS_DATA`` has no metric.py.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from conftest import DIMENSIONS, FAKE_CLASSES

from galaxus.data import metric as metric_module
from galaxus.data.metric import (
    METRIC_MODULE_NAME,
    clear_metric_cache,
    confusions,
    evaluate,
    load_metric,
    score,
)
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
    """The module caches each loaded metric by path; every test starts with an empty cache."""
    clear_metric_cache()
    yield
    clear_metric_cache()


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
def test_parity_the_report_is_indexable_by_dimension_the_way_its_consumers_index_it():
    """The test above checks the names appear *somewhere* in ``str(report)``; this checks the access path.

    ``bench/ocr_language.py`` reads ``evaluate(...)[dimension]["f1_macro"]`` (#101 item 9, PR #108) to compare an
    OCR route against zero-shot, and #106 turns numbers like that one into the per-dimension pick. Nothing in the
    repo records that the real metric.py's report is shaped that way: the coverage test above passes for any shape
    whose string happens to hold the names, and this lane's own doubles return ``{"overall": ...}``
    (``STUB``) and ``{"per_dimension": ..., "overall": ...}`` (``test_saturday_chain``) - neither of which
    supports that access. So this is the one place where the real file settles it.

    If this fails, the shape is not what those consumers assume, and any per-dimension number printed through
    that path needs re-reading before it decides anything.
    """
    from galaxus.data.classes import dimensions

    _, solution, submission, _, _ = _real_holdout_and_submission()
    report = evaluate(solution, submission)

    for name in dimensions():
        assert name in report, f"evaluate()'s report is not keyed by dimension; its keys are {sorted(report)}"
        assert "f1_macro" in report[name], f"{name} carries no f1_macro; its keys are {sorted(report[name])}"
        assert isinstance(report[name]["f1_macro"], int | float)


@_needs_real_data
def test_parity_unscored_cells_are_skipped_dimension_by_dimension():
    """The sharp version, and the one that decides whether our local score matches the leaderboard.

    Changing predictions *inside* unscored cells can, where the class distribution is balanced, move the penalty
    from one class to another and leave the mean untouched - so that check alone can pass against a ``metric.py``
    that wrongly counts unscored cells. This one reads one dimension's own f1 twice: once over every row, once
    with that dimension's unscored rows removed outright. If nulls are skipped the two are identical; if they are
    counted, they differ.

    The dimension cannot be isolated by slicing columns, which is how this test was first written. The real
    metric.py indexes ``[[id_column, *DIMENSIONS]]`` and raises on a frame holding only one of them; the synthetic
    double this ran against until the data arrived was more permissive, so the mistake stayed invisible. Rows are
    dropped instead, and the one dimension is read out of ``evaluate``, whose per-dimension keys the parity test
    above pins. The other dimensions move when rows go - only this one's number is compared.
    """
    _, solution, submission, names, _ = _real_holdout_and_submission()
    checked = 0
    for name in names:
        unscored = solution[name].isna()
        if not unscored.any() or unscored.all():
            continue
        checked += 1
        with_nulls = evaluate(solution, submission)[name]["f1_macro"]
        kept = solution[~unscored.to_numpy()].reset_index(drop=True)
        kept_submission = submission[submission["image_id"].isin(kept["image_id"])].reset_index(drop=True)
        without = evaluate(kept, kept_submission)[name]["f1_macro"]
        assert without == with_nulls, (
            f"{name}: scoring it over every row gives {with_nulls}, but dropping its unscored rows gives "
            f"{without}. metric.py is counting cells our solution leaves empty, so our local score and the "
            "leaderboard's will not agree"
        )
    assert checked > 0, "no dimension had a mix of scored and unscored cells, so this proves nothing"


# --- #73: the cache is per file, not per process ------------------------------------------------------------
#
# A single global cached the first metric.py forever. With the real data present that made 12 failures and 2
# errors in one run (#68): tests that pointed $GALAXUS_DATA at a stub were handed the real metric's classes.


def _stub_metric(folder: Path, value: float) -> Path:
    """A folder holding a metric.py whose ``score`` returns ``value``. Never the real file."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metric.py").write_text(
        "DIMENSIONS = ('subject',)\n"
        f"def score(solution, submission, id_column):\n    return {value}\n"
        f"def evaluate(solution, submission, id_column):\n    return {{'overall': {{'f1_macro': {value}}}}}\n",
        encoding="utf-8",
    )
    return folder


def test_pointing_the_data_dir_at_another_folder_loads_that_folders_metric(tmp_path, monkeypatch):
    first = _stub_metric(tmp_path / "one", 0.25)
    second = _stub_metric(tmp_path / "two", 0.75)

    monkeypatch.setenv(GALAXUS_DATA_ENV, str(first))
    assert load_metric().score(None, None, "image_id") == 0.25

    monkeypatch.setenv(GALAXUS_DATA_ENV, str(second))
    assert load_metric().score(None, None, "image_id") == 0.75, (
        "the second $GALAXUS_DATA was ignored and the first metric came back from the cache (#73)"
    )

    # Back to the first: served from its own cache entry, still the first stub's answer.
    monkeypatch.setenv(GALAXUS_DATA_ENV, str(first))
    assert load_metric().score(None, None, "image_id") == 0.25


def test_the_two_folders_give_two_distinct_modules(tmp_path, monkeypatch):
    first = _stub_metric(tmp_path / "one", 0.25)
    second = _stub_metric(tmp_path / "two", 0.75)
    monkeypatch.setenv(GALAXUS_DATA_ENV, str(first))
    one = load_metric()
    monkeypatch.setenv(GALAXUS_DATA_ENV, str(second))
    assert load_metric() is not one


def test_one_data_dir_still_reads_the_file_once(tmp_path, monkeypatch):
    """Saturday's case: one process, one $GALAXUS_DATA. The file is read and executed once, not per call."""
    folder = tmp_path / "once"
    folder.mkdir()
    (folder / "metric.py").write_text(
        "from pathlib import Path\n"
        "with Path(__file__).with_name('loads.txt').open('a', encoding='utf-8') as handle:\n"
        "    handle.write('x')\n"
        "DIMENSIONS = ('subject',)\n"
        "def score(solution, submission, id_column):\n    return 0.5\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(GALAXUS_DATA_ENV, str(folder))
    assert load_metric() is load_metric()
    assert (folder / "loads.txt").read_text(encoding="utf-8") == "x", "the module was executed more than once"


# --- Emptying the cache -------------------------------------------------------------------------------------


def test_clear_metric_cache_forces_a_re_read_of_the_same_file(tmp_path, monkeypatch):
    folder = _stub_metric(tmp_path / "same", 0.25)
    monkeypatch.setenv(GALAXUS_DATA_ENV, str(folder))
    first = load_metric()
    assert load_metric() is first
    clear_metric_cache()
    assert load_metric() is not first, "clearing the cache must make the next call read the file again"


def test_clearing_an_already_empty_cache_is_harmless():
    clear_metric_cache()
    clear_metric_cache()
    assert metric_module._loaded == {}


def test_assigning_none_still_empties_the_cache(tmp_path, monkeypatch):
    """``tests/serve/conftest.py`` empties the cache by assigning ``None`` to the private global (#68, #77).

    That was how the single global was cleared. Since #73 the cache is a dict, so a bare ``.get`` on ``None``
    would raise ``AttributeError`` -- and the fixture would silently stop isolating the serve tests from a real
    ``$GALAXUS_DATA``, which is the one thing it exists to do. ``None`` therefore means "nothing cached".
    """
    first_folder = _stub_metric(tmp_path / "first", 0.25)
    monkeypatch.setenv(GALAXUS_DATA_ENV, str(first_folder))
    first = load_metric()

    metric_module._loaded = None  # exactly what that fixture does

    second_folder = _stub_metric(tmp_path / "second", 0.75)
    monkeypatch.setenv(GALAXUS_DATA_ENV, str(second_folder))
    reloaded = load_metric()
    assert reloaded is not first
    assert reloaded.score(None, None, "image_id") == 0.75
    assert isinstance(metric_module._loaded, dict), "the cache is a dict again, not left as None"


def test_assigning_none_then_loading_the_same_path_re_reads_it(tmp_path, monkeypatch):
    """The isolation the serve fixture promises: after emptying, nothing is served from before."""
    folder = _stub_metric(tmp_path / "only", 0.5)
    monkeypatch.setenv(GALAXUS_DATA_ENV, str(folder))
    first = load_metric()
    metric_module._loaded = None
    assert load_metric() is not first


# --- #92: confusion pairs per dimension ----------------------------------------------------------------------
#
# A mean macro F1 says a dimension scores badly without saying which classes cause it. These cover the four
# acceptance criteria: scored cells only, no class name typed in the code under test, a dimension with no errors,
# and a prediction outside the allowed classes counted as it came.

DIM = "subject"
OTHER = "background"


def _solution(rows: list[tuple[str, object]]) -> pd.DataFrame:
    """A solution frame: ``None`` marks a cell the metric does not score."""
    return pd.DataFrame({"image_id": [r[0] for r in rows], DIM: [r[1] for r in rows]})


def _submission(rows: list[tuple[str, object]]) -> pd.DataFrame:
    return pd.DataFrame({"image_id": [r[0] for r in rows], DIM: [r[1] for r in rows]})


def test_a_wrong_answer_is_counted_as_the_pair_it_is():
    found = confusions(_solution([("i1", "alpha")]), _submission([("i1", "beta")]), dimension_names=(DIM,))
    assert found[DIM].pairs == (("alpha", "beta", 1),)
    assert found[DIM].wrong == 1
    assert found[DIM].correct == 0
    assert found[DIM].scored == 1


def test_a_correct_answer_is_counted_separately_and_is_not_a_pair():
    found = confusions(_solution([("i1", "alpha")]), _submission([("i1", "alpha")]), dimension_names=(DIM,))
    assert found[DIM].pairs == ()
    assert found[DIM].correct == 1


def test_an_unscored_cell_is_never_counted():
    """solution_frame writes None into the cells metric.py skips, so they cannot reach a count here."""
    solution = _solution([("i1", None), ("i2", "alpha")])
    submission = _submission([("i1", "beta"), ("i2", "alpha")])
    found = confusions(solution, submission, dimension_names=(DIM,))
    assert found[DIM].scored == 1
    assert found[DIM].pairs == ()
    assert found[DIM].correct == 1


def test_a_dimension_with_no_errors_gives_an_empty_list():
    solution = _solution([("i1", "alpha"), ("i2", "beta")])
    found = confusions(solution, _submission([("i1", "alpha"), ("i2", "beta")]), dimension_names=(DIM,))
    assert found[DIM].pairs == ()
    assert found[DIM].wrong == 0


def test_a_class_outside_the_allowed_ones_is_counted_as_it_came():
    """A model inventing a class name is exactly what this table is for, so it is not dropped."""
    found = confusions(_solution([("i1", "alpha")]), _submission([("i1", "not_a_real_class")]), dimension_names=(DIM,))
    assert found[DIM].pairs == (("alpha", "not_a_real_class", 1),)


def test_the_pairs_come_largest_first():
    solution = _solution([(f"i{n}", "alpha") for n in range(5)])
    submission = _submission([("i0", "beta"), ("i1", "beta"), ("i2", "beta"), ("i3", "gamma"), ("i4", "gamma")])
    pairs = confusions(solution, submission, dimension_names=(DIM,))[DIM].pairs
    assert pairs == (("alpha", "beta", 3), ("alpha", "gamma", 2))


def test_pairs_of_equal_count_are_ordered_by_name_so_the_table_is_stable():
    solution = _solution([("i1", "alpha"), ("i2", "alpha")])
    submission = _submission([("i1", "zeta"), ("i2", "beta")])
    assert [p[1] for p in confusions(solution, submission, dimension_names=(DIM,))[DIM].pairs] == ["beta", "zeta"]


def test_a_cell_with_no_answer_at_all_is_missing_rather_than_wrong():
    solution = _solution([("i1", "alpha"), ("i2", "alpha")])
    found = confusions(solution, _submission([("i1", None), ("i2", "")]), dimension_names=(DIM,))[DIM]
    assert found.missing == 2
    assert found.wrong == 0
    assert found.pairs == ()


def test_a_dimension_in_only_one_of_the_frames_is_skipped():
    solution = pd.DataFrame({"image_id": ["i1"], DIM: ["alpha"], OTHER: ["white"]})
    submission = pd.DataFrame({"image_id": ["i1"], DIM: ["beta"]})
    found = confusions(solution, submission, dimension_names=(DIM, OTHER))
    assert DIM in found
    assert OTHER not in found


def test_every_dimension_of_the_contract_is_counted_by_default(monkeypatch):
    """With no dimension_names the contract decides, so no dimension list is typed in the wrapper."""
    monkeypatch.setattr(metric_module, "dimensions", lambda: (DIM, OTHER))
    solution = pd.DataFrame({"image_id": ["i1"], DIM: ["alpha"], OTHER: ["white"]})
    submission = pd.DataFrame({"image_id": ["i1"], DIM: ["beta"], OTHER: ["color"]})
    assert sorted(confusions(solution, submission)) == sorted([DIM, OTHER])


# --- the command line ----------------------------------------------------------------------------------------


@pytest.fixture
def exp_folder(galaxus_data):
    """$GALAXUS_DATA with a splittable train.csv and one EXP's holdout.jsonl, all synthetic."""
    rows = []
    for product in range(20):
        for shot in range(2):
            index = product * 2 + shot
            row = {"image_id": f"t_{index:03d}", "product_id": f"p_{product:02d}"}
            for position, dimension in enumerate(DIMENSIONS):
                row[dimension] = FAKE_CLASSES[dimension][(index + position) % 2]
                row[f"{dimension}_source"] = "human"
                row[f"{dimension}_relevant"] = "false" if (index + position) % 5 == 0 else "true"
            rows.append(row)
    train = pd.DataFrame(rows)
    train.to_csv(galaxus_data / "train.csv", index=False)

    # A classes source, as any real $GALAXUS_DATA has: tests/conftest.py hides contracts/classes.json, so
    # without these the command cannot resolve the dimensions and would exit 1 before counting anything.
    pd.DataFrame({"image_id": train["image_id"], **{d: FAKE_CLASSES[d][0] for d in DIMENSIONS}}).to_csv(
        galaxus_data / "sample_submission.csv", index=False
    )
    (galaxus_data / "metric.py").write_text(
        f"""DIMENSIONS = {tuple(DIMENSIONS)!r}
VALID_CLASSES = {dict(FAKE_CLASSES)!r}
""",
        encoding="utf-8",
    )

    folder = galaxus_data / "out" / "predictions" / "EXP-0042"
    folder.mkdir(parents=True)
    # Always the first class of each dimension, so roughly half the scored cells are wrong.
    (folder / "holdout.jsonl").write_text(
        "\n".join(
            json.dumps({"image_id": image_id, "labels": {d: FAKE_CLASSES[d][0] for d in DIMENSIONS}})
            for image_id in train["image_id"]
        ),
        encoding="utf-8",
    )
    return folder


def test_the_cli_prints_pairs_and_never_an_image_id(exp_folder, capsys):
    assert metric_module.main(["--predictions", str(exp_folder)]) == 0
    printed = capsys.readouterr().out
    assert "scored" in printed and "->" in printed
    assert "nothing written" in printed
    assert "t_000" not in printed and "t_019" not in printed


def test_the_cli_top_limits_the_rows_per_dimension(exp_folder, capsys):
    metric_module.main(["--predictions", str(exp_folder), "--dimensions", DIMENSIONS[0], "--top", "1"])
    printed = capsys.readouterr().out
    assert len([line for line in printed.splitlines() if "->" in line]) == 1


def test_the_cli_refuses_a_top_below_one(exp_folder):
    with pytest.raises(SystemExit):
        metric_module.main(["--predictions", str(exp_folder), "--top", "0"])


def test_the_cli_reports_missing_predictions_without_a_traceback(exp_folder, capsys):
    assert metric_module.main(["--predictions", str(exp_folder.parent / "EXP-9999")]) == 1
    assert capsys.readouterr().err.startswith("cannot work out the confusions:")


def test_a_broken_metric_py_is_not_cached_so_the_next_call_fails_too(galaxus_data):
    """``test_a_broken_metric_py_is_not_left_half_imported`` checks ``sys.modules``; this checks the cache.

    ``_loaded[key] = module`` runs only after the file has executed. If a change ever moved it before, a broken
    file would be cached: the ``sys.modules`` assertion in that other test would still pass, and the *second*
    call would hand back a half-initialised module instead of raising. Verified by making that change - the
    whole file passed without this test, and fails with it.
    """
    (galaxus_data / "metric.py").write_text("raise ValueError('broken')\n", encoding="utf-8")
    for _ in range(2):
        with pytest.raises(ValueError, match="broken"):
            load_metric()
    assert metric_module._loaded == {}
