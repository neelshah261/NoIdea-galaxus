"""#220 (umbrella #214): SigLIP 2 so400m (``galaxus.model.encoders``): the pin, the device, one load per process, the cache, the fused
order, the heads, the ``apertus-core`` base switch and the job's ``encoders`` mode.

Fake encoders and a fake torch; the few images are generated inside the tests. No model, no network.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from galaxus.model import apertus_core, encoders, helper
from galaxus.model.eval import PREDICTORS
from galaxus.types import ImageItem

SHA = "0123456789abcdef0123456789abcdef01234567"
PIN = {encoders.REVISION_ENV: SHA}
ROOT = Path(__file__).resolve().parents[2]


def test_the_pin_comes_from_the_code_and_an_override_must_be_a_full_sha():
    assert encoders.so400m_revision({}) == encoders.PINS[encoders.SO400M_512]
    assert encoders.so400m_revision({encoders.NAME_ENV: encoders.SO400M_384}) == encoders.PINS[encoders.SO400M_384]
    assert all(encoders.SHA.match(pin) for pin in encoders.PINS.values()) and set(encoders.PINS) == set(encoders.CHOICES)
    assert encoders.so400m_revision(PIN) == SHA
    for bad in ({encoders.REVISION_ENV: "main"}, {encoders.REVISION_ENV: SHA[:12]}):
        with pytest.raises(ValueError, match="full commit sha"):
            encoders.so400m_revision(bad)
    assert encoders.so400m_name({}) == encoders.SO400M_512
    assert encoders.so400m_name({encoders.NAME_ENV: encoders.SO400M_384}) == encoders.SO400M_384
    with pytest.raises(ValueError, match="must be one of"):
        encoders.so400m_name({encoders.NAME_ENV: "google/siglip2-base-patch16-224"})
    assert encoders.revision_of(encoders.SO400M_512, PIN) == SHA
    assert encoders.revision_of("facebook/dinov2-base", PIN) == helper.REVISIONS["facebook/dinov2-base"]


def fake_torch(cuda=False, bf16=False, mps=False):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda, is_bf16_supported=lambda: bf16),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        bfloat16="bf16", float16="fp16", float32="fp32",
    )


@pytest.mark.parametrize(("flags", "expected"), [
    ({"cuda": True, "bf16": True}, ("cuda", "bf16")),
    ({"cuda": True, "bf16": False}, ("cuda", "fp16")),  # a T4: no bf16
    ({"mps": True}, ("mps", "fp16")),
    ({}, ("cpu", "fp32")),
])
def test_the_encoder_goes_to_the_best_device_in_a_dtype_it_handles(flags, expected):
    assert encoders.device_and_dtype(fake_torch(**flags)) == expected


class CountingEncoder:
    def __init__(self):
        self.images = 0

    def encode(self, images):
        self.images += len(images)
        return np.tile(np.arange(1, 5, dtype=np.float32), (len(images), 1)) * (self.images or 1)


def pictures(tmp_path, count, start=0):
    out = []
    for k in range(start, start + count):
        path = tmp_path / f"src_{k}.png"
        Image.fromarray(np.full((40, 40, 3), k * 9 % 255, dtype=np.uint8)).save(path)
        out.append((f"img_{k}", str(path)))
    return out


def test_so400m_is_loaded_once_and_its_vectors_land_in_the_shared_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXUS_DATA", str(tmp_path))
    monkeypatch.setattr(encoders, "_LOADED", {})
    loads, encoder = [], CountingEncoder()

    def loader(name, revision):
        loads.append((name, revision))
        return encoder

    for batch in range(3):  # what the live runner does: new images in every batch
        got = encoders.so400m_embeddings(pictures(tmp_path, 4, start=4 * batch), environ=PIN, loader=loader)
        assert len(got) == 4
    assert loads == [(encoders.SO400M_512, SHA)]
    assert encoder.images == 12
    folder = tmp_path / "cache" / "embeddings"
    assert len(list(folder.rglob("*.npy"))) == 12  # the data lane's cache, keyed by the encoder's name
    again = encoders.so400m_embeddings(pictures(tmp_path, 4), environ=PIN, loader=loader)
    assert encoder.images == 12 and set(again) == {f"img_{k}" for k in range(4)}  # all cache hits


def test_the_fused_set_is_exp_0059s_three_then_so400m_each_at_unit_length(monkeypatch):
    widths = {**{name: k + 2 for k, name in enumerate(helper.FUSED_ENCODERS)}, encoders.SO400M_512: 7}

    def vectors(pairs, name):
        return {i: np.random.default_rng(sum(map(ord, i + name))).normal(0, 3, widths[name]) for i, _ in pairs}

    monkeypatch.setattr(helper, "embeddings", lambda pairs, name, **kw: vectors(pairs, name))
    monkeypatch.setattr(encoders, "so400m_embeddings",
                        lambda pairs, environ=None, loader=None: vectors(pairs, encoders.SO400M_512))
    embed, names = encoders.fused_so400m(PIN)
    assert names == (*helper.FUSED_ENCODERS, encoders.SO400M_512)
    got = embed([("a", "x")])["a"]
    start = 0
    for name in names:
        block = got[start : start + widths[name]]
        expected = vectors([("a", "")], name)["a"]
        assert np.allclose(block, expected / np.linalg.norm(expected))
        start += widths[name]
    assert start == len(got)


def labelled_rows(n=160, seed=0):
    rng = np.random.default_rng(seed)
    rows = pd.DataFrame({"image_id": [f"i{k}" for k in range(n)], "image_file": "unused",
                         "product_id": [f"p{k // 2}" for k in range(n)], "kind": rng.choice(["p", "q"], n)})
    rows["kind_source"] = "human"
    return rows


@pytest.mark.parametrize("fused", [False, True])
def test_the_heads_and_the_exp_row_name_so400m_and_its_pin(monkeypatch, fused):
    rows = labelled_rows()
    kind = dict(zip(rows["image_id"], rows["kind"], strict=True))
    rng = np.random.default_rng(1)

    def vectors(pairs, name=None, **kw):
        return {i: ({"p": 3.0, "q": -3.0}[kind[i]] + rng.normal(0, 1, 5)).astype(np.float32) for i, _ in pairs}

    monkeypatch.setattr(helper, "embeddings", lambda pairs, name, **kw: vectors(pairs))
    monkeypatch.setattr(encoders, "so400m_embeddings", lambda pairs, environ=None, loader=None: vectors(pairs))
    data = SimpleNamespace(dimensions=lambda: ("kind",), scored_mask=lambda f, d: f[f"{d}_source"] == "human")
    predictor = encoders.factory(fused=fused, environ=PIN)(rows, data, 0)
    answers = predictor.predict([ImageItem(image_id=i, path="u") for i in rows["image_id"]])
    assert np.mean([a.labels["kind"] == kind[a.image_id] for a in answers]) > 0.95
    assert predictor.helpers[-1] == encoders.SO400M_512
    assert SHA in str(predictor.model_revision)
    assert predictor.heads_model.meta["so400m_revision"] == SHA
    if fused:
        assert predictor.helpers[:3] == helper.FUSED_ENCODERS
        assert predictor.model_revision.startswith(helper.REVISIONS[helper.FUSED_ENCODERS[0]])


def test_the_so400m_variants_are_registered_for_the_eval_script():
    assert {"helper-so400m", "helper-fused-so400m", "apertus-core"} <= set(PREDICTORS)


def test_the_cli_refuses_a_malformed_pin_before_reading_any_data(monkeypatch):
    monkeypatch.setenv(encoders.REVISION_ENV, "main")
    monkeypatch.delenv("GALAXUS_DATA", raising=False)  # it must stop before it looks for the data
    with pytest.raises(ValueError, match="full commit sha"):
        encoders.main(["embed", "--part", "train"])


# ---------------------------------------------------------------- apertus-core's base switch


def test_apertus_core_takes_its_base_from_the_environment():
    embed, names, choice = apertus_core.base_from_env({})
    assert (embed, names, choice) == (helper.fused_embeddings, tuple(helper.FUSED_ENCODERS), "fused")
    assert apertus_core.base_from_env({apertus_core.BASE_ENV: "none"}) == (None, (), "none")
    _, names, _ = apertus_core.base_from_env({apertus_core.BASE_ENV: "fused-so400m", **PIN})
    assert names == (*helper.FUSED_ENCODERS, encoders.SO400M_512)
    _, names, _ = apertus_core.base_from_env({apertus_core.BASE_ENV: "so400m", **PIN})
    assert names == (encoders.SO400M_512,)
    with pytest.raises(ValueError, match="APERTUS_CORE_BASE"):
        apertus_core.base_from_env({apertus_core.BASE_ENV: "large"})


# ---------------------------------------------------------------- the GPU job's encoders mode


def test_the_job_embeds_so400m_in_any_mode_with_the_pin_and_serves_it_apart():
    job = (ROOT / "train" / "hf_job_apertus_features.sh").read_text()
    start = (ROOT / "train" / "hf_job_apertus_features_start.sh").read_text()
    assert "encoders) PARTS=(train dev)" in job
    assert "python -m galaxus.model.encoders embed" in job and "embeddings.tar" in job
    assert "encoders) PARTS=(train dev); SO400M=1" in job  # the encoders mode always embeds so400m
    assert "-e SO400M=1" in start and "-e HELPER_SO400M_REVISION=" in start  # plain variables, not secrets
    assert "embeddings.tar" in start and "encoders)" in start


# ---------------------------------------------------------------- a new batch with an unreadable image


class FakeEncoder:
    def encode(self, images):
        return np.ones((len(images), 3), dtype=np.float32)


def test_so400m_embedding_skips_an_unreadable_image(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXUS_DATA", str(tmp_path))
    monkeypatch.setattr(encoders, "_LOADED", {})
    pairs = []
    for k in range(4):
        path = tmp_path / f"src_{k}.png"
        if k == 1:
            path.write_bytes(b"not an image at all")
        else:
            Image.new("RGB", (20, 20), (k * 40, 0, 0)).save(path)
        pairs.append((f"img_{k}", str(path)))
    logs = []
    embedded = encoders.embed_part(pairs, "csv-new", chunk=4, environ={}, loader=lambda n, r: FakeEncoder(),
                                   log=logs.append)
    assert embedded == 3
    assert len(list((tmp_path / "cache" / "embeddings").rglob("*.npy"))) == 3
    assert any("1 image(s) could not be read" in line for line in logs)
    assert not any("img_1" in line for line in logs)
