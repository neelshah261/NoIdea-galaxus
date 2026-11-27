"""Issue #48: mirror TTA for perspective. A fake inner predictor and synthetic images: no network, no competition data.

The view classes are invented. The overlay dimension and its no-text class are read from the real contract at
runtime, never typed (AGENTS.md "Hard rules"), so this module asks for it with ``real_contract`` (tests/conftest.py).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from galaxus.data.classes import no_overlay_class, overlay_dimension
from galaxus.data.images import cached_image, cached_jpeg, prepared_jpeg
from galaxus.model import eval as model_eval
from galaxus.model.contract import ImageItem, Prediction
from galaxus.model.mirror_tta import MirrorTTA

pytestmark = pytest.mark.real_contract

VIEW, OTHER = "view", "other"
LEFT, RIGHT, FRONT = "view_left", "view_right", "view_front"
TEXT, NO_TEXT, SOME_TEXT = overlay_dimension(), no_overlay_class(), "overlay_invented"
ALLOWED = {
    VIEW: frozenset({LEFT, RIGHT, FRONT}),
    TEXT: frozenset({NO_TEXT, SOME_TEXT}),
    OTHER: frozenset({"other_a", "other_b"}),
}


def honest(red_left: bool) -> tuple[str, float | None]:
    return (LEFT if red_left else RIGHT), None


class FakeInner:
    """Looks at the image it is given: ``view(red on the left edge)`` gives its view class and confidence.

    The other dimension also follows the left edge, so a mirror answers it differently from the original.
    """

    model = "fake-model"
    image_size = 64

    def __init__(self, view=honest, text: str = NO_TEXT) -> None:
        self.view = view
        self.text = text
        self.calls: list[list[ImageItem]] = []

    def predict(self, batch):
        self.calls.append(list(batch))
        predictions = []
        for item in batch:
            with Image.open(item.path) as image:
                red_left = image.convert("RGB").getpixel((0, image.height // 2))[0] > 128
            label, confidence = self.view(red_left)
            predictions.append(
                Prediction(
                    image_id=item.image_id,
                    labels={VIEW: label, TEXT: self.text, OTHER: "other_a" if red_left else "other_b"},
                    confidence=None if confidence is None else {VIEW: confidence, OTHER: 0.5},
                )
            )
        return predictions

    def config(self):
        return {"backend": "fake"}


@pytest.fixture
def item(tmp_path, monkeypatch):
    """A product photo, red on its left half and blue on its right; the image cache lives in a synthetic data dir."""
    (tmp_path / "data").mkdir()
    monkeypatch.setenv("GALAXUS_DATA", str(tmp_path / "data"))
    image = Image.new("RGB", (128, 64), (0, 0, 255))
    image.paste((255, 0, 0), (0, 0, 64, 64))
    image.save(tmp_path / "img_1.png")
    return ImageItem(image_id="img_1", path=tmp_path / "img_1.png", context={"title": "A product"})


def test_the_mirror_is_the_data_lanes_and_an_agreeing_answer_is_kept(item):
    inner = FakeInner(honest)
    tta = MirrorTTA(inner, ALLOWED)

    [prediction] = tta.predict([item])

    assert prediction.labels == {VIEW: LEFT, TEXT: NO_TEXT, OTHER: "other_a"}
    original, mirror = inner.calls
    assert original == [item]
    assert mirror == [
        ImageItem("img_1.mirror", cached_image("img_1", item.path, mirrored=True, image_size=64), item.context)
    ]
    # Read through the cache by its own id, as the zero-shot predictor reads images, it is the prepared mirror itself.
    assert cached_jpeg(mirror[0].image_id, mirror[0].path, image_size=64) == prepared_jpeg(
        item.path, mirrored=True, image_size=64
    )
    assert tta.stats == {"images": 1, "skipped_text": 0, "agreements": 1, "disagreements": 0, "flips": 0}


def test_a_class_without_a_partner_agrees_with_itself(item):
    tta = MirrorTTA(FakeInner(lambda red_left: (FRONT, None)), ALLOWED)

    assert tta.predict([item])[0].labels[VIEW] == FRONT
    assert tta.stats["agreements"] == 1


def test_a_more_confident_mirror_wins_and_every_other_dimension_stays_the_originals(item):
    # The inner always says left: the mirror's "left" is the original's right, and the mirror is surer.
    tta = MirrorTTA(FakeInner(lambda red_left: (LEFT, 0.6 if red_left else 0.9)), ALLOWED)

    [prediction] = tta.predict([item])

    assert prediction.labels == {VIEW: RIGHT, TEXT: NO_TEXT, OTHER: "other_a"}
    assert prediction.confidence == {VIEW: 0.9, OTHER: 0.5}
    assert tta.stats == {"images": 1, "skipped_text": 0, "agreements": 0, "disagreements": 1, "flips": 1}


@pytest.mark.parametrize(
    "confidence",
    [
        pytest.param(lambda red_left: 0.9 if red_left else 0.6, id="original surer"),
        pytest.param(lambda red_left: 0.7, id="tie"),
        pytest.param(lambda red_left: None, id="no confidences"),
        pytest.param(lambda red_left: None if red_left else 0.99, id="only the mirror has one"),
    ],
)
def test_otherwise_a_disagreement_keeps_the_original(item, confidence):
    inner = FakeInner(lambda red_left: (LEFT, confidence(red_left)))
    tta = MirrorTTA(inner, ALLOWED)

    [prediction] = tta.predict([item])

    assert prediction.labels == {VIEW: LEFT, TEXT: NO_TEXT, OTHER: "other_a"}
    assert tta.stats == {"images": 1, "skipped_text": 0, "agreements": 0, "disagreements": 1, "flips": 0}


def test_an_image_with_text_is_not_mirrored(item):
    inner = FakeInner(lambda red_left: (LEFT, 0.1), text=SOME_TEXT)
    tta = MirrorTTA(inner, ALLOWED)

    [prediction] = tta.predict([item])

    assert prediction.labels[VIEW] == LEFT
    assert inner.calls == [[item]]
    assert tta.stats == {"images": 1, "skipped_text": 1, "agreements": 0, "disagreements": 0, "flips": 0}


def test_config_is_the_inner_config_plus_the_counts(item):
    tta = MirrorTTA(FakeInner(), ALLOWED)
    tta.predict([item])

    assert tta.config() == {
        "backend": "fake",
        "mirror_tta": {"dimension": VIEW, "images": 1, "skipped_text": 0, "agreements": 1, "disagreements": 0,
                       "flips": 0},
    }
    assert tta.model == "fake-model"


@pytest.mark.parametrize(
    "allowed",
    [
        pytest.param({**ALLOWED, VIEW: frozenset({FRONT})}, id="no left/right classes"),
        pytest.param({**ALLOWED, TEXT: frozenset({SOME_TEXT})}, id="no no-text class"),
    ],
)
def test_a_class_list_it_cannot_mirror_is_refused(allowed):
    with pytest.raises(ValueError):
        MirrorTTA(FakeInner(), allowed)


def test_registered_in_eval_around_the_inner_predictor(monkeypatch):
    inner = FakeInner()
    monkeypatch.setitem(model_eval.PREDICTORS, "zero-shot", lambda rows, data, seed: inner)

    built = model_eval.PREDICTORS["mirror-tta:zero-shot"](None, SimpleNamespace(allowed_classes=lambda: ALLOWED), 0)

    assert isinstance(built, MirrorTTA)
    assert built.inner is inner
