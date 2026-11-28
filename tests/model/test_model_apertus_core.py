"""#214: ``apertus-core`` (heads on Apertus hidden states with the frozen encoders), the fp32 vision tokenizer of the
local backend, and the GPU feature job's scripts.

Synthetic features in a real ``FeatureStore`` under ``tmp_path``; invented dimensions and classes. No model, no torch,
no network.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from galaxus.model import apertus_core, heads
from galaxus.model.apertus.local import keep_vision_tokenizer_fp32
from galaxus.model.eval import PREDICTORS
from galaxus.types import ImageItem

ROOT = Path(__file__).resolve().parents[2]
HIDDEN = 6


def store_with(tmp_path, vectors, part="train"):
    store = heads.FeatureStore(tmp_path / "store")
    ids = list(vectors)
    store.append(part, ids, np.stack([vectors[i] for i in ids]))
    return store


def test_each_apertus_layer_is_put_at_unit_length_in_order():
    vector = np.concatenate([np.full(HIDDEN, 3.0), np.arange(1, HIDDEN + 1, dtype=float)])
    got = apertus_core.per_layer_unit(vector, 2)
    first, second = got[:HIDDEN], got[HIDDEN:]
    assert np.linalg.norm(first) == pytest.approx(1.0) and np.linalg.norm(second) == pytest.approx(1.0)
    assert np.allclose(second, np.arange(1, HIDDEN + 1) / np.linalg.norm(np.arange(1, HIDDEN + 1)))


def test_the_store_is_read_once_and_reread_when_a_new_part_lands(tmp_path):
    store = store_with(tmp_path, {"a": np.ones(2 * HIDDEN), "b": np.full(2 * HIDDEN, 2.0)})
    features = apertus_core.ApertusFeatures(store, (-1, -9))
    assert set(features.get(["a", "b"])) == {"a", "b"}
    with pytest.raises(KeyError, match="1 image") as missing:
        features.get(["a", "final_1"])
    assert "final_1" not in str(missing.value)  # counts only, never an image id (AGENTS.md rule 3)
    store.append("final", ["final_1"], np.ones((1, 2 * HIDDEN)))  # the 13:00 tar, untarred under the cache
    assert np.linalg.norm(features.get(["final_1"])["final_1"][:HIDDEN]) == pytest.approx(1.0)


def test_the_store_and_layers_come_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXUS_DATA", str(tmp_path))
    features = apertus_core.features_from_env({"APERTUS_CORE_LAYERS": "-1,-9,-17"})
    assert features.layer_ids == (-1, -9, -17) and features.model == apertus_core.APERTUS_8B
    assert features.store.root == heads.store_for(apertus_core.APERTUS_8B, (-1, -9, -17), "last").root
    assert apertus_core.features_from_env({}).layer_ids == (-1, -9)
    assert apertus_core.parse_layers("") == (-1, -9)


def test_base_scores_round_trip_and_refuse_bad_input(tmp_path):
    path = apertus_core.save_base_scores(tmp_path / "out" / "ft.npz", ["x", "y"], [[0.1, -2.0], [-0.5, -0.3]])
    scores = apertus_core.BaseScores(path)
    assert np.allclose(scores.get(["y"])["y"], [-0.5, -0.3])
    with pytest.raises(ValueError, match="one row per image"):
        apertus_core.save_base_scores(tmp_path / "bad.npz", ["x"], [[1.0], [2.0]])
    with pytest.raises(ValueError, match="twice"):
        apertus_core.save_base_scores(tmp_path / "bad.npz", ["x", "x"], [[1.0], [2.0]])
    with pytest.raises(ValueError, match="NaN"):
        apertus_core.save_base_scores(tmp_path / "bad.npz", ["x"], [[np.nan]])
    with pytest.raises(KeyError, match="1 image"):
        scores.get(["z"])
    assert not (tmp_path / "bad.npz").exists()


def test_the_features_are_apertus_then_the_encoders_then_the_scores(tmp_path):
    store = store_with(tmp_path, {"a": np.arange(2 * HIDDEN, dtype=float) + 1})
    scores = apertus_core.BaseScores(apertus_core.save_base_scores(tmp_path / "s.npz", ["a"], [[7.0, 8.0]]))
    embed = apertus_core.core_embeddings(apertus_core.ApertusFeatures(store, (-1, -9)),
                                         base=lambda pairs: {i: np.array([5.0, 5.0, 5.0]) for i, _ in pairs},
                                         scores=scores)
    got = embed([("a", "unused")])["a"]
    assert len(got) == 2 * HIDDEN + 3 + 2
    assert np.allclose(got[:2 * HIDDEN], apertus_core.per_layer_unit(np.arange(2 * HIDDEN) + 1.0, 2))
    assert np.allclose(got[2 * HIDDEN:], [5.0, 5.0, 5.0, 7.0, 8.0])


def test_the_heads_learn_from_the_apertus_block(tmp_path):
    """Only Apertus' block carries the class here, so a good answer proves the core reaches the heads."""
    rng = np.random.default_rng(0)
    n = 160
    rows = pd.DataFrame({"image_id": [f"i{k}" for k in range(n)], "image_file": "unused",
                         "product_id": [f"p{k // 2}" for k in range(n)], "kind": rng.choice(["p", "q"], n)})
    rows["kind_source"] = "human"
    centre = {"p": 4.0, "q": -4.0}
    vectors = {i: centre[k] + rng.normal(0, 1, 2 * HIDDEN) for i, k in zip(rows["image_id"], rows["kind"], strict=True)}
    features = apertus_core.ApertusFeatures(store_with(tmp_path, vectors), (-1, -9), model="apertus-test")
    noise = {i: rng.normal(0, 1, 3) for i in rows["image_id"]}
    data = SimpleNamespace(dimensions=lambda: ("kind",), scored_mask=lambda f, d: f[f"{d}_source"] == "human")
    build = apertus_core.factory(apertus=features, base=lambda pairs: {i: noise[i] for i, _ in pairs},
                                 base_name="google/siglip-base-patch16-224", environ={})
    predictor = build(rows, data, 0)
    answers = predictor.predict([ImageItem(image_id=i, path="u") for i in rows["image_id"]])
    accuracy = np.mean([a.labels["kind"] == k for a, k in zip(answers, rows["kind"], strict=True)])
    assert accuracy > 0.95
    # The EXP row: Apertus classifies, the encoder is its helper, and the meta says how the features were made.
    assert predictor.model == "apertus-test"
    assert predictor.helpers == ("google/siglip-base-patch16-224",)
    meta = predictor.heads_model.meta
    assert (meta["core"], meta["layers"], meta["pool"], meta["base_scores"]) == ("apertus-test", [-1, -9], "last", None)


def test_apertus_core_is_registered_for_the_eval_script():
    assert "apertus-core" in PREDICTORS


# ---------------------------------------------------------------- the local backend's vision tokenizer (#214)


class Module:
    def __init__(self):
        self.floated = False

    def float(self):
        self.floated = True
        return self


def test_the_vision_tokenizer_is_kept_in_float32():
    tokenizer, layer = Module(), Module()
    network = SimpleNamespace(named_modules=lambda: [("model.vision_tokenizer", tokenizer), ("model.layers.0", layer)])
    assert keep_vision_tokenizer_fp32(network) == 1
    assert tokenizer.floated and not layer.floated
    assert keep_vision_tokenizer_fp32(SimpleNamespace()) == 0  # a stand-in without named_modules


# ---------------------------------------------------------------- the GPU feature job's scripts


JOB = ROOT / "train" / "hf_job_apertus_features.sh"
START = ROOT / "train" / "hf_job_apertus_features_start.sh"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("script", [JOB, START], ids=["job", "start"])
def test_the_job_scripts_are_valid_bash(script):
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_the_job_passes_layers_with_an_equals_sign():
    """argparse reads ``--layers -1,-9`` as two options and fails; ``--layers=-1,-9`` works."""
    text = JOB.read_text()
    assert '"--layers=$LAYERS"' in text
    assert not re.search(r"--layers\s+-", text)


def test_the_features_leave_the_job_only_through_its_protected_port():
    job, start = JOB.read_text(), START.read_text()
    assert "http.server 8000" in job and "--expose 8000" in start
    assert "hf upload" not in job and "upload" not in re.sub(r"#.*", "", job)  # no repo, no dataset
    for secret in ("HF_TOKEN", "KAGGLE_API_TOKEN", "PASSPHRASE"):
        assert f"--secrets {secret}" in start  # by name only: the values never enter the command line
    assert "echo \"$PASSPHRASE" not in job and "print(os.environ[\"PASSPHRASE\"]" not in job
