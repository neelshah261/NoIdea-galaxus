"""#175: rows built on another run's saved answers record that run's model (they read it from its EXP row)."""

from __future__ import annotations

import dataclasses

from galaxus.model.recalibrate import RecalibratedPredictor
from galaxus.model.type_prior import TypePriorPredictor


def test_both_predictors_carry_a_model_and_revision_field_for_the_eval_row():
    for cls in (TypePriorPredictor, RecalibratedPredictor):
        names = {f.name for f in dataclasses.fields(cls)}
        assert {"model", "model_revision"} <= names, cls.__name__


class FakeInner:
    model, model_revision = "apertus-v1-5-70b", "rev-1"

    def predict(self, batch):
        return []


def fake_inner():
    return FakeInner()


def test_the_live_paths_record_the_inner_predictors_model(tmp_path, monkeypatch):
    import json

    from galaxus.model import recalibrate, type_prior

    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"table": {}, "rules": {}, "base": "EXP-0051"}), encoding="utf-8")
    for module in (type_prior, recalibrate):
        monkeypatch.setenv(module.RULES_ENV, str(rules))
        monkeypatch.setenv(module.INNER_ENV, "test_model_base_model:fake_inner")
        built = module.from_env()
        assert (built.model, built.model_revision) == ("apertus-v1-5-70b", "rev-1"), module.__name__
