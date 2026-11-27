"""galaxus.types is the one contract, and every lane resolves to it."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from galaxus.types import ImageItem, Prediction, Predictor


def test_image_item_fields_and_missing_context():
    item = ImageItem(image_id="img_a", path=Path("a.jpg"))
    assert [f.name for f in dataclasses.fields(ImageItem)] == ["image_id", "path", "context"]
    assert item.context is None


def test_prediction_fields_and_optional_confidence():
    prediction = Prediction(image_id="img_a", labels={"dim": "cls"})
    assert [f.name for f in dataclasses.fields(Prediction)] == ["image_id", "labels", "confidence"]
    assert prediction.confidence is None


@pytest.mark.parametrize("value", [ImageItem("img_a", "a.jpg"), Prediction("img_a", {"dim": "cls"})])
def test_both_records_are_frozen(value):
    with pytest.raises(dataclasses.FrozenInstanceError):
        value.image_id = "img_b"


def test_any_object_with_predict_is_a_predictor():
    class Echo:
        def predict(self, batch):
            return [Prediction(item.image_id, {"dim": "cls"}) for item in batch]

    assert isinstance(Echo(), Predictor)
    assert not isinstance(object(), Predictor)


def test_the_model_lane_stand_in_resolves_to_these_classes():
    from galaxus.model import contract

    assert contract.ImageItem is ImageItem
    assert contract.Prediction is Prediction
    assert contract.Predictor is Predictor


def test_the_serve_lane_builds_these_items():
    from galaxus.serve.items import image_item_type

    assert image_item_type() is ImageItem
