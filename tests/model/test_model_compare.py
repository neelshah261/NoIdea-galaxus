"""#106: ``galaxus.model.compare``, which variant works with which model. Synthetic rows only, no data, no network."""

from __future__ import annotations

import json

import pytest

from galaxus.model.compare import dimensions, findings, main, select, table, variant


def row(exp_id, predictor, model, scores, mean, rate, **config):
    return {"id": exp_id, "predictor": predictor, "model": model, "config": config, "macro_f1": scores, "score": mean,
            "images_per_second": rate}


ROWS = [
    row("EXP-0001", "majority", None, {"d1": 0.10, "d2": 0.20}, 0.15, 1_805_822.4),
    row("EXP-0002", "zero-shot", "m-8b", {"d1": 0.30, "d2": 0.40}, 0.35, 4.9, context=True),
    row("EXP-0014", "zero-shot-image-first", "m-8b", {"d1": 0.50, "d2": 0.30}, 0.40, 4.8, image_first=True),
    row("EXP-0030", "zero-shot", "m-70b", {"d1": 0.40, "d2": 0.60}, 0.50, 1.2, context=True),
]


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, "default"),
        ({"context": True, "image_size": 512}, "default"),
        ({"context": False}, "no context"),
        ({"drop_context": ["product_title"], "macro_aware": True}, "without product_title, macro-aware"),
        ({"image_first": True, "image": "crop", "image_size": 768}, "image first, crop, 768 px"),
        ({"observe_first": True, "image": "plain"}, "observe first"),
    ],
)
def test_the_variant_is_named_from_the_rows_config(config, expected):
    assert variant(config) == expected


def test_the_table_bolds_the_best_of_each_column_and_hides_the_baselines_speed():
    text = table(ROWS)
    lines = text.splitlines()
    assert lines[0] == "| EXP | predictor | model | variant | mean | d1 | d2 | img/s |"
    assert len(lines) == 2 + len(ROWS)
    image_first = next(line for line in lines if line.startswith("| EXP-0014"))
    seventy = next(line for line in lines if line.startswith("| EXP-0030"))
    assert "**0.500** | 0.300" in image_first  # best on d1, not on d2
    assert "| **0.500** | 0.400 | **0.600** | 1.2 |" in seventy  # best mean and best d2, at 1.2 img/s
    assert "n/a (no model)" in next(line for line in lines if line.startswith("| EXP-0001"))
    assert "1805822" not in text


def test_the_findings_name_the_winners_and_respect_the_speed_floor():
    text = findings(ROWS, floor=2.0)
    assert "- d1: EXP-0014 (m-8b, image first) 0.500" in text
    assert "- d2: EXP-0030 (m-70b, default) 0.600" in text
    assert "Best mean: EXP-0030 (m-70b, default) 0.500" in text
    assert "Best mean at 2 img/s or more: EXP-0014 (m-8b, image first) 0.400, 4.8 img/s" in text
    assert "Best on m-8b: EXP-0014" in text and "Best on m-70b: EXP-0030" in text


def test_select_keeps_the_asked_order_and_refuses_unknown_ids():
    assert [r["id"] for r in select(ROWS, ["EXP-0030", "EXP-0002"])] == ["EXP-0030", "EXP-0002"]
    assert [r["id"] for r in select(ROWS)] == [r["id"] for r in ROWS]
    with pytest.raises(KeyError, match="EXP-0099"):
        select(ROWS, ["EXP-0002", "EXP-0099"])


def test_rows_without_scores_are_left_out_and_dimensions_keep_their_order():
    rows = [*ROWS, {"id": "EXP-0050", "predictor": "lora", "config": {}}]
    assert [r["id"] for r in select(rows)] == [r["id"] for r in ROWS]
    assert dimensions(ROWS) == ["d1", "d2"]


def test_the_command_reads_a_results_file_and_prints_markdown(tmp_path, capsys):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(ROWS), encoding="utf-8")
    assert main(["EXP-0002", "EXP-0014", "--results", str(path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("| EXP | predictor | model | variant |")
    assert "Best per dimension" in out and "EXP-0030" not in out
    assert main(["EXP-0099", "--results", str(path)]) == 2
    assert "EXP-0099" in capsys.readouterr().err
