"""A helper image model picks the class (galaxus.model.helper), with a fake encoder. No model, no data, no network."""

from __future__ import annotations

import numpy as np
from model_synthetic import FAKE_CLASSES, FakeMetric, fake_data_lane, train_table

from galaxus.model import eval as model_eval
from galaxus.model import helper
from galaxus.model.eval import items_from_rows


def label_embeddings(table):
    """Embeddings that carry each image's labels (one block per dimension) plus a little noise."""
    rng = np.random.default_rng(0)
    vectors = {}
    for row in table.to_dict(orient="records"):
        blocks = [(1.0, 0.0) if row[d] == names[0] else (0.0, 1.0) for d, names in FAKE_CLASSES.items()]
        flat = np.array([value for block in blocks for value in block], dtype=np.float32)
        vectors[str(row["image_id"])] = flat + rng.normal(0, 0.05, flat.size).astype(np.float32)
    return vectors


def test_the_helper_learns_each_dimension_from_its_embeddings(tmp_path):
    train = train_table(products=30)
    data = fake_data_lane(train, tmp_path / "out", FakeMetric())
    rows, holdout = data.split_holdout(train)
    truth = label_embeddings(train)
    calls = []

    def embed(pairs, encoder_name):
        pairs = list(pairs)
        calls.append(len(pairs))
        return {image_id: truth[image_id] for image_id, _ in pairs}

    predictor = helper.factory(embed=embed)(rows, data, 0)
    assert calls == [len(rows)]  # every training image embedded in one call
    answers = predictor.predict(items_from_rows(holdout))
    assert calls[-1] == len(holdout)  # the batch's missing embeddings in one call, not one per image
    expected = holdout.set_index("image_id")
    for d in FAKE_CLASSES:
        right = sum(a.labels[d] == expected.loc[a.image_id, d] for a in answers)
        assert right / len(answers) > 0.9, d
    assert predictor.model == helper.DEFAULT_ENCODER and predictor.helpers == (helper.DEFAULT_ENCODER,)


def test_embeddings_come_from_the_cache_and_only_missing_ones_are_computed(tmp_path, monkeypatch):
    computed = []

    def fake_embed_rows(pairs, encoder, encoder_name, directory=None):
        for image_id, _ in pairs:
            computed.append(image_id)
            np.save(directory / f"{image_id}.npy", np.full(4, len(computed), dtype=np.float16))
        return {}

    monkeypatch.setattr("galaxus.data.neighbours.embed_rows", fake_embed_rows)
    loads = []

    def load(name):  # a stub encoder loader
        loads.append(name)
        return object()

    first = helper.embeddings([("a", "/a.jpg"), ("b", "/b.jpg")], load_encoder=load, directory=tmp_path)
    again = helper.embeddings([("a", "/a.jpg"), ("b", "/b.jpg"), ("c", "/c.jpg")], load_encoder=load, directory=tmp_path)
    assert computed == ["a", "b", "c"] and loads == [helper.DEFAULT_ENCODER] * 2
    assert np.array_equal(first["a"], again["a"]) and again["c"].dtype == np.float32
    helper.embeddings([("a", "/a.jpg")], load_encoder=load, directory=tmp_path)
    assert len(loads) == 2  # nothing missing: the encoder is not even loaded


def test_calibration_folds_are_grouped_by_product(tmp_path):
    rows = train_table(products=20)
    pairs = list(helper.product_folds(seed=0, n_folds=5)(rows))
    assert len(pairs) == 5
    assert sorted(i for _, held in pairs for i in held["image_id"]) == sorted(rows["image_id"])
    for fit, held in pairs:
        assert not set(fit["product_id"]) & set(held["product_id"])


def test_registered_in_the_eval_script():
    assert "helper-siglip" in model_eval.PREDICTORS
