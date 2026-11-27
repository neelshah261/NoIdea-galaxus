"""Issue #12: the contract the model lane codes against (PLAN.md "Lanes and contracts", B0 issue #11)."""

from __future__ import annotations

import importlib
import importlib.util

from galaxus.model import contract
from galaxus.model.contract import ImageItem, Prediction


def test_an_image_item_has_the_plan_fields():
    item = ImageItem(image_id="img_1", path="/x.jpg", context=None)
    assert (item.image_id, item.path, item.context) == ("img_1", "/x.jpg", None)


def test_a_prediction_has_the_plan_fields():
    prediction = Prediction(image_id="img_1", labels={"subject": "a"}, confidence=None)
    assert (prediction.image_id, dict(prediction.labels), prediction.confidence) == ("img_1", {"subject": "a"}, None)


def test_the_design_types_are_used_once_they_exist():
    if importlib.util.find_spec("galaxus.types") is None:
        assert contract.ImageItem.__module__ == "galaxus.model.contract"
    else:
        types = importlib.import_module("galaxus.types")
        assert (contract.ImageItem, contract.Prediction, contract.Predictor) == (
            types.ImageItem,
            types.Prediction,
            types.Predictor,
        )
