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


def _method_rows():
    gateway = dict(ROW, id="EXP-0049", model="apertus-v1-5-70b", model_revision=None, score=0.47,
                   config={"prompt_tokens": 9_526_164, "completion_tokens": 238_221, "images": 2444},
                   hardware={"system": "Darwin", "machine": "arm64", "gpu": None})
    prior = dict(ROW, id="EXP-0051", model=None, score=0.49, config={"base": "EXP-0049"})
    siglip = dict(ROW, id="EXP-0055", model="google/siglip-base-patch16-224", model_revision="7fd15f0",
                  helpers=["google/siglip-base-patch16-224"], score=0.705, config={})
    method = dict(ROW, id="EXP-0056", model=None, score=0.733, hardware={"system": "Darwin", "machine": "arm64"},
                  config={"members": {"siglip": "EXP-0055", "70b": "EXP-0051"}, "base": "siglip"})
    other = dict(ROW, id="EXP-0004", model="apertus-lora", score=0.343, config={})
    return [gateway, prior, siglip, method, other], method


def test_the_final_methods_models_come_from_its_members_recursively_not_every_row():
    rows, method = _method_rows()
    assert [r["id"] for r in report.method_rows(rows, method)] == ["EXP-0056", "EXP-0055", "EXP-0051", "EXP-0049"]
    text = report.section7(rows, method, [], None)
    runtime = next(line for line in text.splitlines() if "Runtime:" in line)
    assert "apertus-v1-5-70b" in runtime and "siglip-base-patch16-224 @ 7fd15f0" in runtime
    assert "apertus-lora" not in runtime  # a row outside the method is not a runtime model
    assert "built from: EXP-0055, EXP-0051, EXP-0049" in text


def test_resources_come_from_the_methods_newest_drill_and_its_gateway_tokens():
    rows, method = _method_rows()
    drills = [
        {"exp": "EXP-0056", "predictor": "galaxus.model.ensemble:from_env", "part": "dev", "images": 300,
         "images_per_second": 1.5, "meets_floor": False, "floor_images_per_second": 2.0,
         "finished_at": "2026-11-28T05:00:00+00:00"},
        {"exp": "EXP-0056", "predictor": "galaxus.model.ensemble:from_env", "part": "dev", "images": 300,
         "images_per_second": 1.978, "meets_floor": False, "floor_images_per_second": 2.0,
         "finished_at": "2026-11-28T05:53:00+00:00"},
        {"exp": "EXP-0099", "images_per_second": 50.0, "finished_at": "2026-11-28T09:00:00+00:00"},  # another method
    ]
    text = report.section7(rows, method, [], None, drills=drills)
    for expected in ("**1.978 images/s**", "is BELOW the 2.0 images/s floor", "about 45 min", "Darwin arm64, no GPU",
                     "14.0 hours on that machine", "3,898 prompt and 97 answer tokens per image", "390M prompt tokens"):
        assert expected in text, expected
    assert "50.000" not in text and "galaxus.serve.bench" not in text


def test_development_models_come_from_models_md_and_the_weights_are_written(tmp_path):
    models = tmp_path / "MODELS.md"
    models.write_text("# MODELS\n\n## Runtime\n\n| Model | Role |\n| --- | --- |\n| Apertus | runs |\n\n"
                      "## Development only (never at runtime)\n\n| Model | Role | Where |\n| --- | --- | --- |\n"
                      "| `gpt-6.2-astra` (closed) | reference | laptop |\n| Claude Code (Anthropic) | assistant | all |\n\n"
                      "## Other\n\n| Model | x |\n| --- | --- |\n| not this one | x |\n")
    assert report.development_models(models) == ["`gpt-6.2-astra` (closed)", "Claude Code (Anthropic)"]
    assert report.development_models(tmp_path / "missing.md") == []
    rows, method = _method_rows()
    text = report.section7(rows, method, [], None, development=report.development_models(models),
                           weights="noidea/heads @ abc123")
    assert "Development only, never at runtime (source `MODELS.md`): `gpt-6.2-astra` (closed); Claude Code" in text
    assert "weights: noidea/heads @ abc123." in text


def test_drill_reports_are_read_and_a_broken_one_is_skipped(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "report.json").write_text(json.dumps({"exp": "EXP-0056", "images_per_second": 2.5}))
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "report.json").write_text("{not json")
    assert [d["images_per_second"] for d in report.read_drills(tmp_path)] == [2.5]
    assert report.read_drills(tmp_path / "missing") == []
