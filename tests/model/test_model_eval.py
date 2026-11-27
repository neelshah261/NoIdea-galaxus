"""Issue #12: the eval script. Synthetic tables and a fake metric only: no competition data, no network."""

from __future__ import annotations

import dataclasses
import json

import pytest
from model_synthetic import DIMENSIONS, FAKE_CLASSES, FakeMetric, fake_data_lane, train_table

from galaxus.model import eval as model_eval
from galaxus.model.contract import Prediction
from galaxus.model.eval import DataLane, DirtyCheckout, InvalidPredictions, items_from_rows, main, run_eval
from galaxus.model.results import ResultsFileError, read_rows

COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def clean_checkout(monkeypatch):
    """Every run sees a clean checkout at a fixed commit, unless a test says otherwise."""
    monkeypatch.setattr(model_eval, "git_state", lambda: (COMMIT, False))


@pytest.fixture
def results(tmp_path):
    return tmp_path / "results" / "metrics.json"


def test_a_run_appends_one_exp_row(tmp_path, results):
    row = run_eval("majority", fake_data_lane(train_table(), tmp_path / "out"), results_path=results)
    assert read_rows(results) == [row]
    assert (row["id"], row["predictor"], row["commit"], row["dirty"], row["seed"]) == (
        "EXP-0001",
        "majority",
        COMMIT,
        False,
        0,
    )
    assert row["score"] == 0.25
    assert row["macro_f1"] == dict.fromkeys(DIMENSIONS, 0.25)
    assert row["weighted_f1"] == dict.fromkeys(DIMENSIONS, 0.5)
    assert row["overall_weighted_f1"] == 0.5
    assert (row["holdout"]["images"], row["holdout"]["products"], row["holdout"]["training_images"]) == (6, 2, 24)
    assert len(row["holdout"]["sha256"]) == 16
    assert row["images_per_second"] > 0
    assert row["config"]["answer"] == {dimension: FAKE_CLASSES[dimension][0] for dimension in DIMENSIONS}


def test_ids_count_up_and_are_never_reused(tmp_path, results):
    data = fake_data_lane(train_table(), tmp_path / "out")
    assert run_eval("majority", data, results_path=results)["id"] == "EXP-0001"
    assert run_eval("majority", data, results_path=results)["id"] == "EXP-0002"
    before = results.read_text(encoding="utf-8")
    with pytest.raises(ResultsFileError, match="never reused"):
        run_eval("majority", data, results_path=results, exp_id="EXP-0001")
    assert results.read_text(encoding="utf-8") == before


def test_a_planned_id_can_be_given(tmp_path, results):
    data = fake_data_lane(train_table(), tmp_path / "out")
    assert run_eval("majority", data, results_path=results, exp_id="EXP-0007")["id"] == "EXP-0007"


def test_the_metric_scores_exactly_the_holdout(tmp_path, results):
    metric = FakeMetric()
    train = train_table()
    data = fake_data_lane(train, tmp_path / "out", metric)
    run_eval("majority", data, results_path=results)
    ((solution, submission),) = metric.calls
    _, holdout = data.split_holdout(train)
    assert submission["image_id"].tolist() == holdout["image_id"].tolist()
    assert list(submission.columns) == ["image_id", *DIMENSIONS]
    assert solution["image_id"].tolist() == holdout["image_id"].tolist()
    assert len(submission.drop(columns="image_id").drop_duplicates()) == 1, "the majority answer is one constant row"


def test_a_dirty_checkout_is_refused_unless_allowed(tmp_path, results, monkeypatch):
    monkeypatch.setattr(model_eval, "git_state", lambda: (COMMIT, True))
    data = fake_data_lane(train_table(), tmp_path / "out")
    with pytest.raises(DirtyCheckout):
        run_eval("majority", data, results_path=results)
    assert not results.exists()
    assert run_eval("majority", data, results_path=results, allow_dirty=True)["dirty"] is True


class BrokenPredictor:
    """Answers the first class of every dimension, then lets ``mutate`` break the answers."""

    def __init__(self, mutate):
        self.mutate = mutate

    def predict(self, batch):
        answer = {dimension: FAKE_CLASSES[dimension][0] for dimension in DIMENSIONS}
        return self.mutate([Prediction(image_id=item.image_id, labels=dict(answer)) for item in batch])


def _relabel(predictions, **changes):
    return [Prediction(image_id=p.image_id, labels={**p.labels, **changes}) for p in predictions]


def _drop(predictions, dimension):
    return [Prediction(image_id=p.image_id, labels={k: v for k, v in p.labels.items() if k != dimension}) for p in predictions]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda predictions: predictions[1:], "have no prediction"),
        (lambda predictions: predictions + predictions[:1], "predicted twice"),
        (lambda predictions: predictions + [Prediction(image_id="img_x", labels={})], "not asked for"),
        (lambda predictions: _relabel(predictions, subject="not_a_class"), "is not a class of subject"),
        (lambda predictions: _relabel(predictions, background=""), "no class for background"),
        (lambda predictions: _drop(predictions, "perspective"), "no class for perspective"),
    ],
)
def test_a_predictor_that_breaks_the_contract_writes_nothing(tmp_path, results, monkeypatch, mutate, message):
    monkeypatch.setitem(model_eval.PREDICTORS, "broken", lambda rows, data, seed: BrokenPredictor(mutate))
    with pytest.raises(InvalidPredictions, match=message):
        run_eval("broken", fake_data_lane(train_table(), tmp_path / "out"), results_path=results)
    assert not results.exists()
    assert not (tmp_path / "out").exists()


def test_the_holdout_answers_go_to_the_data_dir(tmp_path, results):
    row = run_eval("majority", fake_data_lane(train_table(), tmp_path / "out"), results_path=results)
    lines = (tmp_path / "out" / "predictions" / row["id"] / "holdout.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == row["holdout"]["images"]
    first = json.loads(lines[0])
    assert set(first) == {"image_id", "labels", "confidence"}
    assert set(first["labels"]) == set(DIMENSIONS)


def test_the_cost_per_100k_images_follows_the_plan_formula(tmp_path, results):
    data = fake_data_lane(train_table(), tmp_path / "out")
    row = run_eval("majority", data, results_path=results, usd_per_hour=1.8)
    assert row["cost_per_100k_usd"] == pytest.approx(1.8 / (row["images_per_second"] * 3600) * 100_000)
    assert run_eval("majority", data, results_path=results)["cost_per_100k_usd"] is None


def test_an_unknown_predictor_is_refused(tmp_path, results):
    with pytest.raises(KeyError, match="unknown predictor"):
        run_eval("no_such_method", fake_data_lane(train_table(), tmp_path / "out"), results_path=results)


def test_the_product_context_goes_into_the_items():
    rows = train_table().head(2).copy()
    rows.loc[rows.index[1], "brand"] = ""
    first, second = items_from_rows(rows)
    assert first.context["product_title"] == rows.iloc[0]["product_title"]
    assert first.path == rows.iloc[0]["image_file"]
    assert "brand" not in second.context


def test_a_row_without_context_or_file_gives_none_and_an_empty_path():
    (item,) = items_from_rows(train_table().head(1)[["image_id"]])
    assert (item.image_id, item.path, item.context) == ("img_00000", "", None)


def test_the_command_line_runs_the_named_predictor(tmp_path, results, monkeypatch, capsys):
    data = fake_data_lane(train_table(), tmp_path / "out")
    monkeypatch.setattr(DataLane, "real", classmethod(lambda cls: data))
    assert main(["majority", "--exp-id", "EXP-0001", "--results", str(results)]) == 0
    printed = capsys.readouterr().out
    assert "EXP-0001" in printed and "subject" in printed and "overall" in printed
    assert [row["id"] for row in read_rows(results)] == ["EXP-0001"]


def test_the_command_line_explains_a_dirty_checkout(tmp_path, results, monkeypatch, capsys):
    monkeypatch.setattr(model_eval, "git_state", lambda: (COMMIT, True))
    data = fake_data_lane(train_table(), tmp_path / "out")
    monkeypatch.setattr(DataLane, "real", classmethod(lambda cls: data))
    assert main(["majority", "--results", str(results)]) == 1
    assert "uncommitted changes" in capsys.readouterr().err
    assert not results.exists()


def test_with_the_real_data_lane(tmp_path, results, monkeypatch):
    """The data lane's loader, split and scored-cell mask on a synthetic train.csv; only metric.py is faked."""
    split = pytest.importorskip("galaxus.data.split", reason="the data lane (#4, #5) is not on this branch yet")
    data_dir = tmp_path / "galaxus-data"
    data_dir.mkdir()
    train_table(products=20).drop(columns=["image_file"]).to_csv(data_dir / "train.csv", index=False)
    monkeypatch.setenv("GALAXUS_DATA", str(data_dir))
    metric = FakeMetric()
    data = dataclasses.replace(
        DataLane.real(),
        dimensions=lambda: DIMENSIONS,
        allowed_classes=lambda: {dimension: frozenset(names) for dimension, names in FAKE_CLASSES.items()},
        score=metric.score,
        evaluate=metric.evaluate,
    )
    row = run_eval("majority", data, results_path=results)
    _, official = split.split_holdout(data.read_train())
    ((_, submission),) = metric.calls
    assert submission["image_id"].tolist() == official["image_id"].tolist()
    assert row["holdout"]["images"] == len(official)
    assert (data_dir / "out" / "predictions" / row["id"] / "holdout.jsonl").is_file()
