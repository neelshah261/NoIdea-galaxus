"""#176, #182: the SigLIP helper's mirror-aware perspective head. Synthetic features, invented classes and a fake
encoder; the one image is generated inside the test. No model, no network.

The features encode the side the way a mirror does: left adds a direction, right subtracts it, and the mirrored
image's features carry the opposite sign.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from galaxus.model import helper
from galaxus.model.eval import PREDICTORS
from galaxus.types import ImageItem

DIMS = ("kind", "view", "overlay")
ALLOWED = {"kind": frozenset({"p", "q"}), "view": frozenset({"x_left", "x_right", "flat"}),
           "overlay": frozenset({"none", "promo"})}
SETUP = helper.MirrorSetup.from_classes(ALLOWED, text_dimension="overlay", no_text="none")
WIDTH = 16


def scored_mask(frame, dimension):
    return frame[f"{dimension}_source"] == "human"


def synthetic(prefix, n_products=150, per=2, seed=0, side_share=0.16):
    rng = np.random.default_rng(seed)
    basis = np.random.default_rng(42).standard_normal((4, WIDTH))  # kind p, kind q, side, promo: fixed across sets
    n = n_products * per
    u = rng.random(n)
    frame = pd.DataFrame({
        "image_id": [f"{prefix}{k}" for k in range(n)],
        "image_file": "unused",
        "product_id": [f"{prefix}p{k // per}" for k in range(n)],
        "kind": rng.choice(["p", "q"], n),
        "view": np.where(u < side_share / 2, "x_left", np.where(u < side_share, "x_right", "flat")),
        "overlay": np.where(rng.random(n) < 0.15, "promo", "none"),
    })
    for d in DIMS:
        frame[f"{d}_source"] = "human"
    side = frame["view"].map({"x_left": 1.0, "x_right": -1.0, "flat": 0.0}).to_numpy()
    kind = np.where(frame["kind"] == "p", 0, 1)
    promo = (frame["overlay"] == "promo").to_numpy(float)
    common = basis[kind] + 2.0 * promo[:, None] * basis[3]
    own = common + 1.1 * side[:, None] * basis[2] + rng.normal(0, 1.0, (n, WIDTH))
    mirrored = common - 1.1 * side[:, None] * basis[2] + rng.normal(0, 1.0, (n, WIDTH))
    ids = frame["image_id"].tolist()
    return frame, dict(zip(ids, own.astype(np.float32))), dict(zip(ids, mirrored.astype(np.float32)))


DATA = SimpleNamespace(dimensions=lambda: DIMS, scored_mask=scored_mask, allowed_classes=lambda: ALLOWED)


def build(mirror, train, own, mirrored):
    def embed(pairs, _name):
        return {i: own[i] for i, _ in pairs}

    def mirror_embed(pairs, _name):
        return {i: mirrored[i] for i, _ in pairs}

    return helper.factory(embed=embed, mirror=mirror, mirror_embed=mirror_embed, setup=SETUP)(train, DATA, 0)


def items(frame):
    return [ImageItem(image_id=i, path="unused") for i in frame["image_id"]]


def macro_f1(truth, answer):
    scores = []
    for c in sorted(set(truth) | set(answer)):
        tp = sum(t == c and a == c for t, a in zip(truth, answer))
        fp = sum(t != c and a == c for t, a in zip(truth, answer))
        fn = sum(t == c and a != c for t, a in zip(truth, answer))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


@pytest.fixture(scope="module")
def trained():
    train, own, mirrored = synthetic("t", seed=0)
    test, test_own, test_mirrored = synthetic("h", seed=9)
    own, mirrored = {**own, **test_own}, {**mirrored, **test_mirrored}
    return train, test, own, mirrored, build(True, train, own, mirrored), build(False, train, own, mirrored)


def test_the_mirror_head_beats_the_plain_head_on_scarce_left_right_data(trained):
    _, test, _, _, with_mirror, plain = trained
    truth = test["view"].tolist()
    scores = {name: macro_f1(truth, [p.labels["view"] for p in model.predict(items(test))])
              for name, model in (("mirror", with_mirror), ("plain", plain))}
    assert scores["mirror"] > scores["plain"], scores


def test_an_image_and_its_mirror_get_swapped_answers(trained):
    _, test, own, mirrored, with_mirror, _ = trained
    row = test[(test["overlay"] == "none") & (test["view"] == "x_left")].iloc[0]
    image, flipped = row["image_id"], row["image_id"] + "_flipped"
    with_mirror.features[flipped], with_mirror.mirror_features[flipped] = mirrored[image], own[image]
    a, b = with_mirror.predict([ImageItem(image_id=image, path="u"), ImageItem(image_id=flipped, path="u")])
    assert b.labels["view"] == SETUP.swap(a.labels["view"])
    assert b.confidence["view"] == pytest.approx(a.confidence["view"])


def test_images_with_text_keep_the_heads_own_answer(trained):
    _, test, own, _, with_mirror, _ = trained
    promo = test[test["overlay"] == "promo"]
    answers = with_mirror.predict(items(promo))
    texty = [(p, own[p.image_id]) for p in answers if p.labels["overlay"] == "promo"]
    assert texty
    head = with_mirror.heads_model.heads["view"]
    expected, _ = head.predict(np.stack([x for _, x in texty]))
    assert [p.labels["view"] for p, _ in texty] == expected


def test_the_eval_row_records_the_mirror_and_how_many_rows_it_added(trained):
    train, _, _, _, with_mirror, _ = trained
    config = with_mirror.config()
    assert config["mirror"]["dimension"] == "view"
    added = config["calibration"]["view"]["mirrored_training_rows"]
    sided_text_free = ((train["view"] != "flat") & (train["overlay"] == "none")).sum()
    assert added == sided_text_free  # exactly the human, text-free, left/right rows


def test_mirrored_embeddings_flip_the_pixels_and_are_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXUS_DATA", str(tmp_path))
    picture = np.zeros((64, 96, 3), dtype=np.uint8)
    picture[:, :48] = 220  # bright left half, dark right half
    source = tmp_path / "img.png"
    Image.fromarray(picture).save(source)
    loaded = []

    class Encoder:
        def encode(self, images):
            out = []
            for image in images:
                a = np.asarray(image, dtype=np.float32).mean(axis=(0, 2))
                out.append([a[: len(a) // 2].mean() - a[len(a) // 2 :].mean()])
            return np.asarray(out)

    def load(name):
        loaded.append(name)
        return Encoder()

    folder = tmp_path / "mirror"
    first = helper.mirrored_embeddings([("img_1", str(source))], load_encoder=load, directory=folder)
    original = helper.embeddings([("img_1", str(source))], load_encoder=load, directory=tmp_path / "plain")
    assert first["img_1"][0] < -100 < 100 < original["img_1"][0]  # the bright half moved to the right
    again = helper.mirrored_embeddings([("img_1", str(source))], load_encoder=load, directory=folder)
    assert loaded == [helper.DEFAULT_ENCODER, helper.DEFAULT_ENCODER]  # once per cache, not again
    assert np.array_equal(again["img_1"], first["img_1"])


def test_the_variant_is_registered_for_the_eval_script():
    assert "helper-siglip-mirror" in PREDICTORS and "helper-siglip" in PREDICTORS
