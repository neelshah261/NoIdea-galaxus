"""#41 heads and #42 calibration on synthetic features. No Apertus, no competition data, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from galaxus.model import heads
from galaxus.model.calibration import apply_bias, fit_bias, macro_f1
from galaxus.model.contract import ImageItem

DIMS = ("subject", "perspective")


def synthetic(n=240, d=16, seed=0, rare=0.1):
    """Features where class identity is linearly encoded; class 'c' is rare."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(["a", "b", "c"], size=n, p=[0.6, 0.3, 0.1] if rare == 0.1 else None)
    centers = {"a": rng.normal(0, 1, d), "b": rng.normal(0, 1, d), "c": rng.normal(0, 1, d)}
    x = np.stack([centers[label] + rng.normal(0, 0.8, d) for label in labels]).astype(np.float32)
    return x, labels


def frame_and_features(n=240):
    x, labels = synthetic(n)
    ids = [f"img_{i:04d}" for i in range(n)]
    frame = pd.DataFrame({"image_id": ids, "product_id": [f"p{i // 2}" for i in range(n)]})
    for dim in DIMS:
        frame[dim] = labels
        frame[f"{dim}_source"] = ["human" if i % 10 else "model" for i in range(n)]
        frame[f"{dim}_relevant"] = "true"
    return frame, {i: v.reshape(1, -1) for i, v in zip(ids, x)}


def scored(rows, dim):
    return rows[f"{dim}_source"] == "human"


def folds(rows):
    order = rows["product_id"].astype("category").cat.codes % 3
    for k in range(3):
        yield rows[order != k], rows[order == k]


def test_prompt_with_and_without_context():
    assert "brand: acme" in heads.prompt_for({"brand": "acme", "product_title": "Mouse"})
    assert heads.prompt_for(None) == heads.prompt_for({}) and "  " not in heads.prompt_for(None)
    assert heads.prompt_for({"product_title": "Mouse"}).index("product title") < heads.prompt_for(
        {"product_title": "Mouse"}).index("Classify")


def test_feature_key_changes_with_every_setting():
    keys = {
        heads.feature_key("swiss-ai/Apertus-v1.5-8B", (-1,), "last"),
        heads.feature_key("swiss-ai/Apertus-v1.5-8B", (-1, -9), "last"),
        heads.feature_key("swiss-ai/Apertus-v1.5-8B", (-1,), "mean"),
        heads.feature_key("swiss-ai/Apertus-v1.5-70B", (-1,), "last"),
    }
    assert len(keys) == 4 and all("512px" in key for key in keys)


def test_extract_is_resumable_and_never_recomputes(tmp_path):
    store = heads.FeatureStore(tmp_path / "features")
    items = [ImageItem(f"img_{i}", f"/x/{i}.jpg", {"brand": "b"}) for i in range(7)]
    calls = []

    def forward(images, prompt):
        calls.append(prompt)
        return np.ones((len(images), 1, 4)) * len(calls)

    assert heads.extract(forward, items[:5], store, "train", lambda item: b"jpeg", shard_size=2, log=None) == 5
    assert heads.extract(forward, items, store, "train", lambda item: b"jpeg", shard_size=2, log=None) == 2
    loaded = store.load("train")
    assert sorted(loaded) == sorted(item.image_id for item in items) and len(calls) == 7
    assert loaded["img_0"].dtype == np.float16 and loaded["img_0"].shape == (4,)
    assert not list((tmp_path / "features" / "train").glob(".*tmp*"))


def test_a_head_learns_linearly_encoded_classes():
    x, labels = synthetic(400)
    head = heads.fit_head(x[:300], labels[:300].tolist())
    predicted, confidence = head.predict(x[300:])
    assert np.mean(np.asarray(predicted) == labels[300:]) > 0.85
    assert ((confidence > 0) & (confidence <= 1)).all()


def test_calibration_never_lowers_macro_f1_on_its_fit_data():
    rng = np.random.default_rng(1)
    true = rng.choice(3, size=500, p=[0.8, 0.15, 0.05])
    log_probs = np.log(np.full((500, 3), 0.05))
    log_probs[np.arange(500), true] += 0.6  # weak signal...
    log_probs[:, 0] += 1.0  # ...drowned by a prior towards the frequent class
    before = macro_f1(true, apply_bias(log_probs, np.zeros(3)))
    bias = fit_bias(log_probs, true)
    after = macro_f1(true, apply_bias(log_probs, bias))
    assert after >= before and after > before + 0.1
    assert fit_bias(log_probs[:0], true[:0]).tolist() == [0, 0, 0]


def test_macro_f1_counts_predicted_classes_that_never_occur():
    assert macro_f1(np.array([0, 0]), np.array([0, 0])) == 1.0
    assert macro_f1(np.array([0, 0]), np.array([0, 1])) < 1.0


def test_fit_heads_with_folds_saves_and_loads_the_same_model(tmp_path):
    frame, features = frame_and_features()
    model = heads.fit_heads(frame, features, DIMS, scored, folds, key="k", log=None)
    assert set(model.heads) == set(DIMS) and model.heads["subject"].bias.shape == (3,)
    path = model.save(tmp_path / "heads.npz")
    again = heads.HeadsModel.load(path)
    x = np.stack([features[i].reshape(-1) for i in frame.image_id[:20]])
    for dim in DIMS:
        assert again.heads[dim].predict(x)[0] == model.heads[dim].predict(x)[0]
    assert again.key == "k"


def test_fit_heads_refuses_rows_without_features():
    frame, features = frame_and_features(40)
    features.pop("img_0003")
    with pytest.raises(KeyError, match="no features"):
        heads.fit_heads(frame, features, DIMS, scored, log=None)


def test_predictor_meets_the_contract_and_computes_missing_features(tmp_path):
    frame, features = frame_and_features()
    model = heads.fit_heads(frame, features, DIMS, scored, log=None)
    known = {k: v for k, v in features.items() if k != "img_0001"}
    computed = []

    def compute(item):
        computed.append(item.image_id)
        return features[item.image_id]

    predictor = heads.HeadsPredictor(model, known, compute)
    items = [ImageItem(i, "") for i in frame.image_id[:5]]
    predictions = predictor.predict(items)
    assert [p.image_id for p in predictions] == list(frame.image_id[:5]) and computed == ["img_0001"]
    assert all(set(p.labels) == set(DIMS) and all(0 < v <= 1 for v in p.confidence.values()) for p in predictions)

    serve_runner = pytest.importorskip("galaxus.serve.runner")
    report = serve_runner.run(predictor, items, "EXP-9999", "dev", root=tmp_path, log=None)
    assert report.complete

    with pytest.raises(KeyError, match="no Apertus backend"):
        heads.HeadsPredictor(model, {}).predict([ImageItem("img_0002", "")])


def test_out_of_fold_predictions_find_a_planted_label_error(tmp_path):
    frame, features = frame_and_features(300)
    # Plant one wrong human label on a typical 'a' image: the heads, never trained on its product, should disagree.
    centre = np.mean([features[i].reshape(-1) for i, lab in zip(frame.image_id, frame.subject) if lab == "a"], axis=0)
    candidates = frame[(frame.subject == "a") & (frame.subject_source == "human")].image_id
    planted = min(candidates, key=lambda i: np.linalg.norm(features[i].reshape(-1) - centre))
    frame.loc[frame.image_id == planted, "subject"] = "b"

    records = heads.out_of_fold_records(frame, features, DIMS, scored, folds)
    assert len(records) == len(frame) and all(set(r["labels"]) == set(DIMS) for r in records)
    path = heads.write_records(records, tmp_path / "oof.jsonl")

    label_errors = pytest.importorskip("galaxus.data.label_errors")
    report = label_errors.find_label_errors(path, frame, threshold=0.5, dimension_names=DIMS)
    assert list(report.columns) == ["image_id", "dimension", "label", "suggested_label"]
    hit = report[(report.image_id == planted) & (report.dimension == "subject")]
    assert len(hit) == 1 and hit.iloc[0]["label"] == "b" and hit.iloc[0]["suggested_label"] == "a"
