"""#132 distillation: a teacher's confident labels fill the LoRA's masked prelabel cells. Synthetic rows only."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from galaxus.model import lora

DIMS = ("subject", "background")

PATH = Path(__file__).resolve().parents[2] / "train" / "lora_apertus.py"
spec = importlib.util.spec_from_file_location("galaxus_train_lora_apertus_teacher", PATH)
la = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = la
spec.loader.exec_module(la)


def rows(n: int = 300) -> pd.DataFrame:
    """Subject is human on every row; background is a model prelabel on the odd rows."""
    records = []
    for i in range(n):
        records.append({"image_id": f"i{i}", "image_file": f"/x/{i}.jpg", "subject": "a", "subject_source": "human",
                        "background": "plain", "background_source": "model" if i % 2 else "human"})
    return pd.DataFrame(records)


def teacher(confidence: float, label: str = "scene", subject: str = "z") -> dict:
    return {f"i{i}": {"image_id": f"i{i}", "labels": {"subject": subject, "background": label},
                      "confidence": {"subject": 0.99, "background": confidence}} for i in range(300)}


def test_a_confident_teacher_fills_the_masked_prelabels_and_never_touches_human_cells():
    data = lora.build_training_set(rows(), DIMS, teacher=teacher(0.9))
    odd = [e for e in data.examples if int(e.image_id[1:]) % 2]
    even = [e for e in data.examples if not int(e.image_id[1:]) % 2]
    assert all(e.target["background"] == "scene" and "background" in e.trained_dimensions for e in odd)
    assert all(e.target["background"] == "plain" for e in even)            # human cells stay human
    assert all(e.target["subject"] == "a" for e in data.examples)          # even where the teacher disagrees
    assert data.stats["teacher_cells"] == {"subject": 0, "background": 150}
    assert data.stats["prelabels_masked"]["background"] == 0


def test_an_unsure_teacher_changes_nothing_and_no_teacher_is_the_old_recipe():
    plain = lora.build_training_set(rows(), DIMS)
    unsure = lora.build_training_set(rows(), DIMS, teacher=teacher(0.5))
    assert [e.target for e in unsure.examples] == [e.target for e in plain.examples]
    assert [e.trained_dimensions for e in unsure.examples] == [e.trained_dimensions for e in plain.examples]
    assert unsure.stats["teacher_cells"]["background"] == 0
    assert plain.stats["prelabels_masked"]["background"] == 150
    lowered = lora.build_training_set(rows(), DIMS, teacher=teacher(0.5), teacher_min_confidence=0.4)
    assert lowered.stats["teacher_cells"]["background"] == 150


def test_images_the_teacher_did_not_answer_keep_their_mask():
    partial = {k: v for k, v in teacher(0.9).items() if k in ("i1", "i3")}
    data = lora.build_training_set(rows(), DIMS, teacher=partial)
    assert data.stats["teacher_cells"]["background"] == 2
    i5 = next(e for e in data.examples if e.image_id == "i5")
    assert "background" not in i5.trained_dimensions


def test_the_flags_default_off():
    defaults = la.build_parser().parse_args([])
    assert defaults.teacher is None and defaults.teacher_min_confidence == lora.TEACHER_MIN_CONFIDENCE


def test_a_teacher_file_with_holdout_images_is_refused(monkeypatch, tmp_path):
    frame = rows(10)
    monkeypatch.setattr("galaxus.data.classes.dimensions", lambda: DIMS)
    monkeypatch.setattr("galaxus.data.classes.allowed_classes", lambda: {d: frozenset() for d in DIMS})
    monkeypatch.setattr("galaxus.data.load.read_train", lambda: frame)
    monkeypatch.setattr("galaxus.data.split.split_holdout", lambda f: (f.iloc[2:], f.iloc[:2]))
    monkeypatch.setattr("galaxus.data.mirror.mirrored_rows", lambda r: r.iloc[0:0])
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(v) for v in teacher(0.9).values() if v["image_id"] in ("i0", "i5")) + "\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match="holdout"):
        la.training_set(str(path))
    clean = tmp_path / "clean.jsonl"
    clean.write_text(json.dumps(teacher(0.9)["i5"]) + "\n", encoding="utf-8")
    _, data, _ = la.training_set(str(clean))
    assert data.stats["teacher_cells"]["background"] == 1
