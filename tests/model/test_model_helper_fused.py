"""#182: ``helper-fused`` (EXP-0059), one shared encoder per process, and the fit-on-every-label option.

Fake encoders and synthetic features; the few images are generated inside the tests. No model, no network.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from galaxus.model import heads, helper
from galaxus.model.eval import PREDICTORS
from galaxus.types import ImageItem

WIDTHS = {"google/siglip-base-patch16-224": 3, "facebook/dinov2-base": 4, "google/siglip2-base-patch16-224": 5}


def per_encoder(pairs, name):
    """A different, fixed vector per image and encoder, with a different width per encoder."""
    out = {}
    for image_id, _ in pairs:
        seed = sum(map(ord, image_id + name))
        out[image_id] = np.random.default_rng(seed).normal(0, 3, WIDTHS[name]).astype(np.float32)
    return out


def test_fused_features_follow_exp_0059s_recipe_bit_for_bit():
    pairs = [("img_1", "a"), ("img_2", "b")]
    got = helper.fused(per_encoder)(pairs)
    for image_id, _ in pairs:
        parts = [per_encoder([(image_id, "")], name)[image_id] for name in helper.FUSED_ENCODERS]
        expected = np.concatenate([v / np.linalg.norm(v) for v in parts])
        assert np.array_equal(got[image_id], expected)
        a, b = 0, 0
        for name in helper.FUSED_ENCODERS:  # each encoder's block is at unit length, in the fixed order
            b += WIDTHS[name]
            assert np.linalg.norm(got[image_id][a:b]) == pytest.approx(1.0, abs=1e-6)
            a = b


def test_the_three_encoders_are_pinned_in_exp_0059s_order():
    assert helper.FUSED_ENCODERS == (
        "google/siglip-base-patch16-224", "facebook/dinov2-base", "google/siglip2-base-patch16-224")
    assert {name: helper.REVISIONS[name] for name in helper.FUSED_ENCODERS} == {
        "google/siglip-base-patch16-224": "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
        "facebook/dinov2-base": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
        "google/siglip2-base-patch16-224": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    }


def test_the_eval_row_names_the_three_encoders_and_their_revisions():
    model = heads.HeadsModel(heads={}, key="helper:fused", meta={})
    predictor = helper.HelperPredictor(model, helper.FUSED_NAME, embed=lambda pairs, name: {})
    assert predictor.helpers == helper.FUSED_ENCODERS
    assert predictor.model_revision == ", ".join(helper.REVISIONS[n] for n in helper.FUSED_ENCODERS)
    single = helper.HelperPredictor(model, helper.DEFAULT_ENCODER, embed=lambda pairs, name: {})
    assert single.model_revision == helper.REVISIONS[helper.DEFAULT_ENCODER]  # unchanged for one encoder


def test_the_fused_factory_fits_the_same_heads_on_the_fused_features():
    rng = np.random.default_rng(0)
    n = 160
    rows = pd.DataFrame({"image_id": [f"i{k}" for k in range(n)], "image_file": "unused",
                         "product_id": [f"p{k // 2}" for k in range(n)], "kind": rng.choice(["p", "q"], n)})
    rows["kind_source"] = "human"
    centre = {"p": 4.0, "q": -4.0}

    def encoder_vectors(pairs, name):
        kind = dict(zip(rows["image_id"], rows["kind"], strict=True))
        return {i: (centre[kind[i]] + rng.normal(0, 1, WIDTHS[name])).astype(np.float32) for i, _ in pairs}

    data = SimpleNamespace(dimensions=lambda: ("kind",), scored_mask=lambda f, d: f[f"{d}_source"] == "human")
    embed = helper.fused(encoder_vectors)
    predictor = helper.factory(encoder_name=helper.FUSED_NAME, embed=embed)(rows, data, 0)
    answers = predictor.predict([ImageItem(image_id=i, path="u") for i in rows["image_id"]])
    accuracy = np.mean([a.labels["kind"] == k for a, k in zip(answers, rows["kind"], strict=True)])
    assert accuracy > 0.95 and predictor.heads_model.key == f"helper:{helper.FUSED_NAME}"


def test_the_variants_are_registered_for_the_eval_script():
    assert {"helper-fused", "helper-fused-mirror", "helper-siglip", "helper-siglip-mirror"} <= set(PREDICTORS)


# ---------------------------------------------------------------- one encoder per process (#181)


class CountingEncoder:
    def __init__(self):
        self.images = 0

    def encode(self, images):
        self.images += len(images)
        return np.ones((len(images), 2), dtype=np.float32)


def pictures(tmp_path, count, start=0):
    out = []
    for k in range(start, start + count):
        path = tmp_path / f"src_{k}.png"
        Image.fromarray(np.full((40, 40, 3), k * 7 % 255, dtype=np.uint8)).save(path)
        out.append((f"img_{k}", str(path)))
    return out


def test_new_images_in_every_batch_load_the_encoder_once(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXUS_DATA", str(tmp_path))
    monkeypatch.setattr(helper, "_ENCODERS", {})
    loads = []
    monkeypatch.setattr(helper, "load_pinned_encoder", lambda name: loads.append(name) or CountingEncoder())
    for batch in range(3):  # what final_run does: every batch brings images nobody has embedded
        helper.embeddings(pictures(tmp_path, 4, start=4 * batch), directory=tmp_path / "emb")
    assert loads == [helper.DEFAULT_ENCODER]


def test_two_members_sharing_an_encoder_embed_each_image_once(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXUS_DATA", str(tmp_path))
    encoder, pairs, start = CountingEncoder(), pictures(tmp_path, 6), threading.Barrier(2)
    results = []

    def member():
        start.wait(timeout=10)
        results.append(helper.embeddings(pairs, load_encoder=lambda name: encoder, directory=tmp_path / "emb"))

    threads = [threading.Thread(target=member) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert encoder.images == len(pairs)  # the second member found every embedding cached
    assert all(np.array_equal(results[0][i], results[1][i]) for i, _ in pairs)


# ---------------------------------------------------------------- fit on every labelled row (#176)


def test_fit_all_uses_every_labelled_row_only_when_asked(monkeypatch):
    train = pd.DataFrame({"image_id": [f"i{k}" for k in range(10)]})
    fake = SimpleNamespace(read_train=lambda: train, split_holdout=lambda frame: (frame.iloc[:7], frame.iloc[7:]))
    monkeypatch.setattr("galaxus.model.eval.DataLane.real", classmethod(lambda cls: fake))
    seen = []

    def build(rows, data, seed):
        seen.append(len(rows))
        return SimpleNamespace(heads_model=SimpleNamespace(meta={}))

    everything = helper._live(build, {"HELPER_FIT_ALL": "1"})
    default = helper._live(build, {})
    assert seen == [10, 7]
    assert "holdout included" in everything.heads_model.meta["fitted_on"]
    assert default.heads_model.meta["fitted_on"] == "train minus the official holdout"


# ---------------------------------------------------------------- C (#207): the larger encoder set


def test_every_encoder_set_is_pinned_and_named_like_its_fused_helper():
    assert helper.ENCODER_SETS["base"] == helper.FUSED_ENCODERS
    for encoders in helper.ENCODER_SETS.values():
        assert all(name in helper.REVISIONS for name in encoders)
        assert helper.COMPONENTS["fused:" + "+".join(encoders)] == encoders
    assert helper.REVISIONS["facebook/dinov2-large"] == "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"
    assert helper.REVISIONS["google/siglip2-large-patch16-256"] == "787800c8990e6f058423089178e718139608408c"
