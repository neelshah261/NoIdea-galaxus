"""#106: every zero-shot variant can be scored (eval) and run (runner, submit, final_run). No data, no network."""

from __future__ import annotations

import pytest

from galaxus.model import eval as model_eval
from galaxus.model import mirror_tta, zero_shot
from galaxus.serve.items import load_predictor


class FakePredictor:
    def predict(self, batch):
        return []


def test_every_variant_has_an_eval_name_and_every_wrappable_one_a_mirror_tta_twin():
    for name in zero_shot.VARIANTS:
        assert name in model_eval.PREDICTORS
        assert (f"mirror-tta:{name}" in model_eval.PREDICTORS) is zero_shot.mirror_tta_can_wrap(name)
    assert "mirror-tta:zero-shot" in model_eval.PREDICTORS  # EXP-0006 keeps its name
    assert not zero_shot.mirror_tta_can_wrap("zero-shot-crop")
    assert not zero_shot.mirror_tta_can_wrap("zero-shot-image-first-crop")


def test_the_earlier_names_keep_their_settings_so_their_rows_stay_comparable():
    assert zero_shot.VARIANTS["zero-shot"] == {}
    assert zero_shot.VARIANTS["zero-shot-no-context"] == {"use_context": False}
    assert zero_shot.VARIANTS["zero-shot-image-first"] == {"image_first": True}
    assert all(zero_shot.VARIANTS[name] == options for name, options in zero_shot.PRESETS.items())


@pytest.mark.parametrize("name", sorted(zero_shot.VARIANTS))
def test_from_preset_builds_the_variant_the_environment_names(monkeypatch, name):
    built = {}
    monkeypatch.setattr(zero_shot, "ZeroShotPredictor", lambda **options: built.update(options) or "built")
    monkeypatch.setenv(zero_shot.PRESET_ENV, name)
    assert zero_shot.from_preset() == "built"
    assert built == zero_shot.VARIANTS[name]


def test_from_preset_defaults_to_zero_shot_and_refuses_an_unknown_name(monkeypatch):
    monkeypatch.delenv(zero_shot.PRESET_ENV, raising=False)
    assert zero_shot.preset_name() == "zero-shot"
    monkeypatch.setenv(zero_shot.PRESET_ENV, "zero-shot-sepia")
    with pytest.raises(ValueError, match="zero-shot-sepia.*zero-shot-crop"):
        zero_shot.preset_name()


def test_mirror_tta_from_env_wraps_the_chosen_variant_and_refuses_a_crop_one(monkeypatch):
    inner = FakePredictor()
    monkeypatch.setattr(zero_shot, "from_preset", lambda: inner)
    monkeypatch.setattr(mirror_tta, "MirrorTTA", lambda wrapped: ("tta", wrapped))
    monkeypatch.setenv(zero_shot.PRESET_ENV, "zero-shot-image-first")
    assert mirror_tta.from_env() == ("tta", inner)
    monkeypatch.setenv(zero_shot.PRESET_ENV, "zero-shot-crop")
    with pytest.raises(ValueError, match="cannot wrap 'zero-shot-crop'"):
        mirror_tta.from_env()


def test_the_runner_loads_both_entry_points_by_name(monkeypatch):
    """What submit and final_run do with ``--run`` / ``--predictor``: a spec, called with no arguments."""
    monkeypatch.setattr(zero_shot, "ZeroShotPredictor", lambda **options: FakePredictor())
    monkeypatch.setattr(mirror_tta, "MirrorTTA", lambda wrapped: FakePredictor())
    monkeypatch.setenv(zero_shot.PRESET_ENV, "zero-shot-macro-aware")
    assert isinstance(load_predictor("galaxus.model.zero_shot:from_preset"), FakePredictor)
    assert isinstance(load_predictor("galaxus.model.mirror_tta:from_env"), FakePredictor)
