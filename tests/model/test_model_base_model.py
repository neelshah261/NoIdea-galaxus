"""#175: rows built on another run's saved answers record that run's model (they read it from its EXP row)."""

from __future__ import annotations

import dataclasses

from galaxus.model.recalibrate import RecalibratedPredictor
from galaxus.model.type_prior import TypePriorPredictor


def test_both_predictors_carry_a_model_and_revision_field_for_the_eval_row():
    for cls in (TypePriorPredictor, RecalibratedPredictor):
        names = {f.name for f in dataclasses.fields(cls)}
        assert {"model", "model_revision"} <= names, cls.__name__
