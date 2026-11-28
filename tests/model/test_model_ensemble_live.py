"""#182: the live ensemble predictor for the final run, and the determinism of the whole combination.

Fake members and invented classes only: no model, no network.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from galaxus.model import ensemble, heads, helper
from galaxus.types import ImageItem, Prediction

DIMS = ("subject", "perspective")
IDS = [f"img_{k:03d}" for k in range(12)]


class Member:
    """A fake live member: a fixed answer per image, and a count of what it was asked."""

    def __init__(self, labels, confidence=0.8, model=None, helpers=()):
        self.labels, self.confidence, self.asked = labels, confidence, []
        self.model, self.helpers = model, helpers

    def predict(self, batch):
        self.asked.extend(str(item.image_id) for item in batch)
        return [Prediction(image_id=str(item.image_id), labels=self.labels(str(item.image_id)),
                           confidence={d: self.confidence for d in DIMS}) for item in batch]

    def config(self):
        return {"fake": True}


def siglip(image_id):
    return {"subject": "s" if int(image_id[-1]) % 2 else "t", "perspective": "l"}


def big(image_id):
    return {"subject": "s", "perspective": "r" if int(image_id[-1]) % 3 else "l"}


CHOICE = {
    "dimensions": list(DIMS),
    "routing": {"subject": "vote", "perspective": "siglip"},
    "weights": {"subject": {"siglip": 0.7, "big": 0.5}, "perspective": {"siglip": 0.6, "big": 0.4}},
    "member_exps": {"siglip": "EXP-0055", "big": "EXP-0051"},
    "rule": "agree",
}


def items(ids=IDS):
    return [ImageItem(image_id=i, path="unused") for i in ids]


def live(cache_dir=None, choice=CHOICE):
    members = {"siglip": Member(siglip, model="google/siglip-base-patch16-224", helpers=("siglip",)),
               "big": Member(big, model="apertus-v1-5-70b")}
    return ensemble.LiveEnsemblePredictor(choice=json.loads(json.dumps(choice)), predictors=members,
                                          cache_dir=cache_dir), members


def records(member):
    return {i: {"image_id": i, "labels": member.labels(i), "confidence": {d: member.confidence for d in DIMS}}
            for i in IDS}


def test_live_answers_equal_the_saved_answer_path_exactly():
    predictor, members = live()
    saved = ensemble.apply(CHOICE, {name: records(m) for name, m in members.items()})
    assert predictor.predict(items()) == saved


def test_a_member_the_routing_never_reads_is_not_needed_but_a_voter_is():
    only_siglip = {**CHOICE, "routing": {"subject": "siglip", "perspective": "siglip"}}
    assert ensemble.needed_members(only_siglip) == ["siglip"]
    assert ensemble.needed_members(CHOICE) == ["big", "siglip"]  # the vote on subject counts both
    with pytest.raises(ValueError, match="no predictor"):
        ensemble.LiveEnsemblePredictor(choice=CHOICE, predictors={"siglip": Member(siglip)})


def test_the_cache_makes_a_rerun_identical_and_asks_each_member_once_per_image(tmp_path):
    first, members = live(tmp_path)
    answers = first.predict(items(IDS[:7]))
    answers += first.predict(items(IDS[7:]))
    assert sorted(members["big"].asked) == IDS

    rerun, fresh_members = live(tmp_path)
    fresh_members["big"].labels = lambda image_id: {"subject": "t", "perspective": "l"}  # a gateway that drifted
    assert rerun.predict(items()) == answers
    assert fresh_members["big"].asked == [] and fresh_members["siglip"].asked == []
    lines = (tmp_path / "big.jsonl").read_text(encoding="utf-8").splitlines()
    assert sorted(json.loads(line)["image_id"] for line in lines) == IDS  # each image once


def test_a_resume_after_a_crash_asks_only_the_images_not_yet_answered(tmp_path):
    first, _ = live(tmp_path)
    first.predict(items(IDS[:5]))
    resumed, members = live(tmp_path)
    resumed.predict(items())
    assert members["big"].asked == IDS[5:]


def test_concurrent_batches_cache_every_image_once(tmp_path):
    predictor, _ = live(tmp_path)
    batches = [IDS[k::3] for k in range(3)]
    threads = [threading.Thread(target=predictor.predict, args=(items(b),)) for b in batches]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    lines = (tmp_path / "siglip.jsonl").read_text(encoding="utf-8").splitlines()
    assert sorted(json.loads(line)["image_id"] for line in lines) == IDS


def test_the_members_answer_at_the_same_time():
    """Each member waits for the other inside predict: this passes only when they run concurrently (no clock)."""
    meeting = threading.Barrier(2)

    class Waiting(Member):
        def predict(self, batch):
            meeting.wait(timeout=10)  # in turn, the first member would wait alone and break the barrier
            return super().predict(batch)

    predictor = ensemble.LiveEnsemblePredictor(choice=CHOICE, predictors={"siglip": Waiting(siglip), "big": Waiting(big)})
    saved = ensemble.apply(CHOICE, {"siglip": records(Member(siglip)), "big": records(Member(big))})
    assert predictor.predict(items()) == saved


def test_a_failing_member_fails_the_batch():
    class Broken(Member):
        def predict(self, batch):
            raise ConnectionError("gateway down")

    predictor = ensemble.LiveEnsemblePredictor(choice=CHOICE, predictors={"siglip": Member(siglip), "big": Broken(big)})
    with pytest.raises(ConnectionError, match="gateway down"):
        predictor.predict(items())


def test_the_eval_row_names_the_members_models_helpers_and_the_frozen_routing():
    predictor, _ = live()
    assert predictor.model == "apertus-v1-5-70b, google/siglip-base-patch16-224"
    assert predictor.helpers == ("siglip",)
    config = predictor.config()
    assert config["routing"] == CHOICE["routing"] and len(config["routing_sha256"]) == 16
    assert set(config["members"]) == {"big", "siglip"}


def test_from_env_builds_only_the_members_the_routing_reads(tmp_path, monkeypatch):
    routing = tmp_path / "routing.json"
    routing.write_text(json.dumps({**CHOICE, "routing": {"subject": "siglip", "perspective": "siglip"}}),
                       encoding="utf-8")
    built = []
    monkeypatch.setattr("galaxus.serve.items.load_predictor",
                        lambda spec: built.append(spec) or Member(siglip))
    environ = {"ENSEMBLE_ROUTING": str(routing), "ENSEMBLE_CACHE": str(tmp_path / "cache"),
               "ENSEMBLE_PREDICTORS": "siglip=galaxus.model.helper:from_env big=galaxus.model.type_prior:from_env"}
    predictor = ensemble.from_env(environ)
    assert built == ["galaxus.model.helper:from_env"] and predictor.cache_dir == tmp_path / "cache"


def test_from_env_names_what_is_missing(tmp_path):
    with pytest.raises(RuntimeError, match="ENSEMBLE_ROUTING"):
        ensemble.from_env({})
    routing = tmp_path / "routing.json"
    routing.write_text(json.dumps(CHOICE), encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"\['big'\]"):
        ensemble.from_env({"ENSEMBLE_ROUTING": str(routing), "ENSEMBLE_PREDICTORS": "siglip=a.b:c"})


# ---------------------------------------------------------------- the SigLIP helper is pinned


def test_the_helper_loads_its_encoder_at_the_pinned_revision(monkeypatch, tmp_path):
    asked = {}

    def snapshot_download(repo_id, revision, allow_patterns):
        asked.update(repo_id=repo_id, revision=revision, files=allow_patterns)
        return str(tmp_path / "snapshot")

    monkeypatch.setitem(sys.modules, "huggingface_hub", type(sys)("huggingface_hub"))
    sys.modules["huggingface_hub"].snapshot_download = snapshot_download
    monkeypatch.setattr("galaxus.data.neighbours.load_encoder", lambda path: ("encoder", path))
    assert helper.load_pinned_encoder(helper.DEFAULT_ENCODER) == ("encoder", str(tmp_path / "snapshot"))
    assert asked["revision"] == "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
    assert "pytorch_model.bin" not in asked["files"] and "model.safetensors" in asked["files"]


def test_an_unpinned_encoder_is_refused_and_the_eval_row_records_the_revision():
    with pytest.raises(ValueError, match="REVISIONS"):
        helper.pinned_revision("someone/unpinned-encoder")
    model = heads.HeadsModel(heads={}, key="helper:test", meta={})
    predictor = helper.HelperPredictor(model, helper.DEFAULT_ENCODER, embed=lambda pairs, name: {})
    assert predictor.model_revision == "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"


# ---------------------------------------------------------------- the whole combination is deterministic

SCRIPT = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[2])
    import numpy as np, pandas as pd
    from test_model_combination import MEMBERS, DIMS, catalog, member_answers, solution_frame, evaluate, scored
    from galaxus.model import ensemble, type_prior

    holdout, rows, dev = catalog("h", seed=0), catalog("t", seed=3), catalog("d", n_products=30, seed=7)
    held, dev_answers = member_answers(holdout), member_answers(dev, seed=5)
    members = {name: held[name] for name in MEMBERS}
    choice = ensemble.choose_by_rule(members, holdout, DIMS, solution_frame, evaluate, ensemble.AGREE)
    combined = {p.image_id: {"image_id": p.image_id, "labels": dict(p.labels), "confidence": dict(p.confidence)}
                for p in ensemble.apply(choice, {name: dev_answers[name] for name in MEMBERS})}
    on_holdout = {p.image_id: {"labels": dict(p.labels), "confidence": dict(p.confidence)}
                  for p in ensemble.apply(choice, members)}
    table = type_prior.type_table(rows, DIMS, scored)
    _, _, full = type_prior.crossfit(on_holdout, holdout, table, DIMS, solution_frame, evaluate)
    final = type_prior.apply_saved({"table": table, **full}, combined, dev)
    out = [{"id": p.image_id, "labels": p.labels, "confidence": p.confidence} for p in final]
    open(sys.argv[1], "w").write(json.dumps({"routing": choice["routing"], "rules": full["rules"], "answers": out},
                                            sort_keys=True))
    """
)


def test_the_combination_gives_byte_identical_files_under_different_hash_seeds(tmp_path):
    """String hashing is randomised per process; nothing in choose, apply or the type prior may depend on it."""
    script = tmp_path / "combine.py"
    script.write_text(SCRIPT, encoding="utf-8")
    outputs = []
    for seed in ("1", "2", "12345"):
        target = tmp_path / f"out_{seed}.json"
        env = {**os.environ, "PYTHONHASHSEED": seed}
        subprocess.run([sys.executable, str(script), str(target), str(Path(__file__).parent)], env=env, check=True,
                       timeout=120)
        outputs.append(target.read_bytes())
    assert outputs[0] == outputs[1] == outputs[2]
