"""#33 speed matrix: the grid, the GPU run sheet, the fake run and the table. No GPU, no network, no data."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "bench" / "speed_matrix.py"
spec = importlib.util.spec_from_file_location("galaxus_bench_speed_matrix", MODULE_PATH)
matrix = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = matrix
spec.loader.exec_module(matrix)


def test_grid_skips_cells_the_local_backend_cannot_have():
    cells = matrix.grid([512], [8, 16], [1, 8], ["vllm", "local"], ["bf16", "fp8"], [True, False])
    local = [c for c in cells if c.backend == "local"]
    assert local and all(c.workers == 1 and not c.prefix_caching and c.precision == "bf16" for c in local)
    assert len([c for c in cells if c.backend == "vllm"]) == 2 * 2 * 2 * 2
    assert len({c.label for c in cells}) == len(cells)


def test_grid_refuses_unknown_settings_and_odd_areas():
    with pytest.raises(ValueError):
        matrix.grid([512], [8], [1], ["tpu"], ["bf16"], [True])
    with pytest.raises(ValueError, match="multiple of 16"):
        matrix.grid([500], [8], [1], ["vllm"], ["bf16"], [True])


def test_plan_starts_each_server_once_and_pins_512_by_default():
    cells = matrix.grid([512], [8, 16], [1, 8], ["vllm"], ["bf16", "fp8"], [True])
    sheet = matrix.plan(cells, "pkg.mod:Pred", 1.8, 128)
    assert sheet.count("bash bench/vllm_apertus.sh") == 2
    assert "MAX_PIXELS=262144" in sheet and "QUANT=fp8" in sheet and "QUANT=none" in sheet
    assert sheet.count("python -m galaxus.serve.bench") == len(cells)
    assert "--dollars-per-hour 1.8" in sheet and "stop the GPU" in sheet


def test_fake_run_and_table(tmp_path):
    cells = matrix.grid([256, 512], [8], [1, 4], ["vllm"], ["bf16"], [True])
    matrix.run_fake(cells, tmp_path, images=32)
    assert matrix.read_results(tmp_path / "bench") == []  # fake rows never count as results
    rows = matrix.read_results(tmp_path / "bench", include_fake=True)
    assert len(rows) == len(cells) and all(r["images_per_second"] > 0 for r in rows)
    text = matrix.table(rows)
    assert text.splitlines()[0].startswith("| Cell | img/s") and len(text.splitlines()) == 2 + len(cells)


def test_table_keeps_the_latest_result_per_cell(tmp_path):
    folder = tmp_path / "bench"
    folder.mkdir()
    rows = [
        {"label": "matrix-vllm-bf16-512px-b8-w1-pc", "images_per_second": 3.0, "finished_at": "2026-11-27T10:00:00"},
        {"label": "matrix-vllm-bf16-512px-b8-w1-pc", "images_per_second": 7.0, "finished_at": "2026-11-27T11:00:00"},
        {"label": "other-run", "images_per_second": 99.0, "finished_at": "2026-11-27T11:00:00"},
    ]
    (folder / "x.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    latest = matrix.read_results(folder)
    assert [r["images_per_second"] for r in latest] == [7.0]
    assert "No matrix results" in matrix.table([])
