"""#52: the pitch deck build. Synthetic EXP rows only: no competition data, no browser."""

from __future__ import annotations

import json
import re

from galaxus.serve import deck

ROWS = {
    "EXP-0002": {"id": "EXP-0002", "predictor": "zero-shot", "model": "m-8b", "score": 0.30341,
                 "macro_f1": {"subject": 0.056, "perspective": 0.097}, "images_per_second": 6.43, "commit": "abc"},
    "EXP-0025": {"id": "EXP-0025", "predictor": "zero-shot-observe-first-no-context", "model": "m-8b", "score": 0.3601,
                 "macro_f1": {"subject": 0.233, "perspective": 0.218}, "images_per_second": 4.62,
                 "cost_per_100k_usd": None, "holdout": {"images": 2444}},
}


def test_fields_are_formatted_and_a_missing_row_or_null_field_is_pending_not_a_number():
    out = deck.fill("{EXP-0025.score}|{EXP-0025.images_per_second}|{EXP-0025.holdout.images}|"
                    "{EXP-0025.cost_per_100k_usd}|{EXP-0099.score}|{EXP-0025.macro_f1.nothing}", ROWS)
    assert out.split("|") == ["0.360", "4.6", "2,444", deck.PENDING, deck.PENDING, deck.PENDING]


def test_best_delta_raw_method_and_label_errors():
    template = "{best.score} {delta EXP-0025.score EXP-0002.score} {delta EXP-0099.score EXP-0002.score} " \
               "{raw best.score} {raw EXP-0099.score} {method} {label_errors}"
    out = deck.fill(template, ROWS, best="EXP-0025", method="Apertus <8B>", label_errors=1234)
    assert out == f"0.360 +0.057 {deck.PENDING} 0.3601 0 Apertus &lt;8B&gt; 1,234"
    assert deck.fill("{best.score} {method} {label_errors}", ROWS) == " ".join([deck.PENDING] * 3)


def test_rows_are_best_first_with_the_gain_over_the_first_id_and_skip_missing_ids():
    out = deck.fill("{rows EXP-0002 EXP-0099 EXP-0025}", ROWS)
    lines = out.splitlines()
    assert len(lines) == 2 and "EXP-0025" in lines[0] and "class=best" in lines[0]
    assert "+0.057" in lines[0] and "+0.000" in lines[1]
    majority = {"EXP-0001": {"id": "EXP-0001", "predictor": "majority", "model": None, "score": 0.187,
                             "images_per_second": 1805822.4}}
    assert "1805822" not in deck.fill("{rows EXP-0001}", majority) and "None" not in deck.fill("{rows EXP-0001}", majority)


def test_pending_inside_an_attribute_does_not_end_it():
    assert '"' not in deck.fill("{best.config.workers}", ROWS)


def test_the_chart_takes_best_and_is_pending_without_rows():
    assert "<svg" in deck.fill("{chart EXP-0002 best}", ROWS, best="EXP-0025")
    assert deck.fill("{chart EXP-0099}", ROWS) == deck.PENDING


def test_the_real_template_has_ten_slides_and_no_placeholder_left_unfilled():
    template = deck.TEMPLATE.read_text(encoding="utf-8")
    assert template.count("<section") == 10  # README.md "Pitch deck": max. 10 slides
    out = deck.fill(template, {})
    assert not re.search(r"\{(EXP-|best|delta|raw|rows|chart|method|label_errors)", out)
    assert out.count("data-speaker-notes=") == 10


def test_main_writes_the_page_outside_the_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXUS_DATA", str(tmp_path / "data"))
    (tmp_path / "data").mkdir()
    results = tmp_path / "metrics.json"
    results.write_text(json.dumps(list(ROWS.values())), encoding="utf-8")
    assert deck.main(["--results", str(results), "--best", "EXP-0025", "--no-pdf"]) == 0
    page = (tmp_path / "data" / "out" / "deck" / "deck.html").read_text(encoding="utf-8")
    assert "0.360" in page and "{best." not in page
