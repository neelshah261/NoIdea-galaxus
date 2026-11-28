"""galaxus.serve.publish: only fitted weights leave, found inside the final predictor. Synthetic heads only."""

from __future__ import annotations

import json

import numpy as np

from galaxus.model.heads import Head, HeadsModel
from galaxus.serve import publish


def _heads(meta=None):
    head = Head(classes=["a", "b", "c"], weights=np.ones((4, 3), np.float32), intercept=np.zeros(3, np.float32),
                mean=np.zeros(4), scale=np.ones(4), bias=np.zeros(3))
    return HeadsModel(heads={"subject": head}, key="helper:x", meta=meta or {"seed": 0})


class Helper:  # what galaxus.model.helper.HelperPredictor exposes
    def __init__(self, encoder, revision):
        self.heads_model = _heads()
        self.encoder_name, self.model_revision = encoder, revision
        self.features = {"img_00001": np.ones(4)}  # cached embeddings: must never be published


class Prior:  # a wrapper with an inner predictor and no heads of its own (the 70B member)
    def __init__(self):
        self.inner = object()


class Ensemble:  # galaxus.model.ensemble.LiveEnsemblePredictor
    def __init__(self):
        self.predictors = {"siglip": Helper("google/siglip-base-patch16-224", "7fd15f0"), "70b": Prior(),
                           "fused": Helper("fused:siglip+dinov2+siglip2", None)}


def test_heads_are_found_inside_the_live_ensemble_and_only_the_weights_are_written(tmp_path):
    found = publish.find_heads(Ensemble())
    assert [(f.name, f.encoder, f.revision) for f in found] == [
        ("fused", "fused:siglip+dinov2+siglip2", None), ("siglip", "google/siglip-base-patch16-224", "7fd15f0")]
    files = publish.write_folder(tmp_path, "EXP-0061", {"score": 0.768}, "abc1234def", found)
    assert sorted(files) == ["fused.heads.npz", "siglip.heads.npz"]
    loaded = HeadsModel.load(tmp_path / "siglip.heads.npz")  # numpy only: allow_pickle=False inside
    assert loaded.heads["subject"].classes == ["a", "b", "c"]
    card = (tmp_path / "README.md").read_text(encoding="utf-8")
    for expected in ("EXP-0061", "0.7680", "abc1234def", "`7fd15f0`", "license: cc-by-4.0", "No image, image id"):
        assert expected in card, expected
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"] == files and manifest["encoders"]["siglip"]["revision"] == "7fd15f0"
    assert publish.check_no_data(tmp_path) == []
    written = b"".join(p.read_bytes() for p in tmp_path.iterdir())
    assert b"img_00001" not in written  # the cached embeddings' ids never reach a file


def test_anything_that_looks_like_an_image_id_blocks_publishing(tmp_path):
    leaky = publish.FoundHeads("siglip", _heads(meta={"seed": 0, "worst": "img_00042"}), "e", "r")
    publish.write_folder(tmp_path, "EXP-0001", None, "c", [leaky])
    problems = publish.check_no_data(tmp_path)
    assert problems and any("img" in p or "image id" in p for p in problems)


def test_a_predictor_without_heads_has_nothing_to_publish():
    assert publish.find_heads(Prior()) == [] and publish.find_heads(None) == []
