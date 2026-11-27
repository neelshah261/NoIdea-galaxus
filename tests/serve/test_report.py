"""Section 7 generator: numbers only from generated files, TODO for everything unmeasured. Synthetic inputs only."""

from __future__ import annotations

import json
import re

from galaxus.serve import report

ROW = {
    "id": "EXP-0007",
    "predictor": "heads",
    "score": 0.81234,
    "overall_weighted_f1": 0.9,
    "macro_f1": {"subject": 0.7, "background": 0.9, "perspective": 0.6, "individuals": 0.99,
                 "text_classification": 0.8, "text_language": 0.88},
    "weighted_f1": {"subject": 0.8},
    "holdout": {"images": 2444},
    "commit": "abcdef1234567890",
    "dirty": False,
    "model": "swiss-ai/Apertus-v1.5-8B",
    "model_revision": "r1",
    "helpers": [],
    "images_per_second": 6.5,
    "cost_per_100k_usd": 7.69,
    "note": "heads | calibrated",
}


def test_full_section_from_generated_files_only():
    bench = [{"label": "matrix-x", "device": "NVIDIA L40S", "images_per_second": 6.5, "peak_memory_gb": 21.0,
              "memory_note": "cuda", "batch_size": 16, "workers": 8, "cost_per_100k": 7.69, "dollars_per_hour": 1.8}]
    text = report.section7([ROW], ROW, bench, label_errors=42, public_score="0.8123", team="NoIdea Zurich")
    assert text.startswith("## 7. Galaxus challenge: results and resources")
    for expected in ("0.812", "0.600", "NVIDIA L40S", "6.50 images/s", "$7.69", "abcdef123456",
                     "42 suspected label errors", "Apertus-v1.5-8B @ r1", "NoIdea Zurich", "0.8123"):
        assert expected in text, expected
    assert "heads / calibrated" in text  # a pipe in a note never breaks the table


def test_nothing_measured_gives_todos_not_numbers():
    text = report.section7([], None, [], None)
    assert text.count("TODO(") >= 6
    assert not re.search(r"\d\.\d{2}", text)  # no score, speed or cost anywhere


def test_pick_named_or_best_and_fake_bench_rows_are_ignored(tmp_path):
    rows = [dict(ROW, id="EXP-0001", score=0.2), dict(ROW, id="EXP-0002", score=0.9)]
    assert report.pick(rows, None)["id"] == "EXP-0002" and report.pick(rows, "EXP-0001")["id"] == "EXP-0001"
    (tmp_path / "a.jsonl").write_text(json.dumps({"label": "fake-x", "device": "cpu"}) + "\n"
                                      + json.dumps({"label": "matrix-y", "device": "gpu"}) + "\n")
    assert [b["label"] for b in report.read_bench(tmp_path)] == ["matrix-y"]


def test_label_error_count_and_kaggle_score_parsing(tmp_path):
    csv_file = tmp_path / "label_errors.csv"
    csv_file.write_text("image_id,dimension,label,suggested_label\na,b,c,d\ne,f,g,h\n")
    assert report.count_label_errors(csv_file) == 2 and report.count_label_errors(tmp_path / "none.csv") is None
    output = "fileName,date,description,status,publicScore,privateScore\nf.csv,2026,x,complete,0.4321,\n"
    assert report.kaggle_public_score(lambda command: output) == "0.4321"
    assert report.kaggle_public_score(lambda command: "fileName,publicScore\n") is None
