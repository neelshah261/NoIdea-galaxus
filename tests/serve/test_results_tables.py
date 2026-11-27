"""#57 results tables and chart from synthetic EXP rows only."""

from __future__ import annotations

import json
import xml.dom.minidom

from galaxus.serve import results_tables as rt

DIMS = ("subject", "background", "perspective", "individuals", "text_classification", "text_language")


def row(exp, score, perspective, sha="h1", parent=None, predictor="heads"):
    macro = {d: score for d in DIMS}
    macro["perspective"] = perspective
    config = {"parent": parent} if parent else {}
    return {"id": exp, "predictor": predictor, "score": score, "macro_f1": macro, "overall_weighted_f1": score + 0.1,
            "images_per_second": 6.0, "cost_per_100k_usd": 8.0, "commit": "abcdef123456", "config": config,
            "holdout": {"sha256": sha}}


ROWS = [row("EXP-0001", 0.20, 0.05, predictor="majority"), row("EXP-0002", 0.70, 0.60),
        row("EXP-0003", 0.75, 0.70, parent="EXP-0001"), row("EXP-0004", 0.80, 0.72, sha="other")]


def test_all_runs_table_has_every_row_and_dimension():
    table = rt.all_runs_table(ROWS)
    assert all(r["id"] in table for r in ROWS) and all(d in table.splitlines()[0] for d in DIMS)
    assert "`abcdef12`" in table


def test_ablation_uses_the_parent_and_never_crosses_holdouts():
    table = rt.ablation_table(ROWS)
    lines = {line.split("|")[1].strip(): line for line in table.splitlines()[2:]}
    assert "| EXP-0001 | config.parent |" in lines["EXP-0003"] and "+0.550" in lines["EXP-0003"]
    assert "| EXP-0001 | previous run, same holdout |" in lines["EXP-0002"] and "+0.500" in lines["EXP-0002"]
    assert "first run on this holdout" in lines["EXP-0004"]
    crossed = rt.ablation_table([ROWS[0], row("EXP-0005", 0.9, 0.9, sha="other", parent="EXP-0001")])
    assert "different holdout" in crossed


def test_perspective_line_names_the_best_run():
    line = rt.perspective_line(ROWS)
    assert "best EXP-0004" in line and "0.720" in line
    assert "no run" in rt.perspective_line([])


def test_chart_is_valid_svg_with_one_bar_per_value():
    svg = rt.chart_svg(ROWS[1:3])
    xml.dom.minidom.parseString(svg)
    assert svg.count("<rect") == 2 + 2 * len(DIMS) + 2  # background + plot, bars, legend swatches
    assert rt.chart_svg([]).startswith("<svg")


def test_cli_writes_markdown_and_chart(tmp_path):
    results = tmp_path / "metrics.json"
    results.write_text(json.dumps(ROWS), encoding="utf-8")
    out, chart = tmp_path / "results.md", tmp_path / "chart.svg"
    assert rt.main(["--results", str(results), "--out", str(out), "--chart", str(chart)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "## What each change did" in text and "chart.svg" in text and "2 different holdouts" in text
    xml.dom.minidom.parse(str(chart))
    empty = tmp_path / "none.json"
    assert rt.main(["--results", str(empty), "--out", str(out), "--chart", str(chart)]) == 0
    assert "No EXP rows" in out.read_text(encoding="utf-8")


def test_only_the_best_run_is_orange_and_the_baseline_is_grey():
    palette = rt.colours([ROWS[0], ROWS[1], ROWS[2]])
    assert palette[2] == rt.ACCENT and palette[0] == rt.RULE2 and palette[1] not in (rt.ACCENT, rt.RULE2)
