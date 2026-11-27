"""B7 numbers and the decryption step, without data or network."""

from __future__ import annotations

import json
import subprocess

import pytest
from serve_helpers import StubPredictor

from galaxus.serve import bench, final_run


def test_cost_formula_matches_plan_md():
    # $/h ÷ (img/s × 3600) × 100,000 with round numbers: 3.6 $/h at 10 img/s → 10 $ per 100k.
    assert bench.cost_per_100k(3.6, 10) == pytest.approx(10.0)
    assert bench.hours_per_100k(10) == pytest.approx(100_000 / 36_000)
    with pytest.raises(ValueError):
        bench.cost_per_100k(1.0, 0)


def test_bench_times_after_warmup_and_saves(items, tmp_path):
    predictor = StubPredictor()
    result = bench.bench(predictor, items, batch_size=1, warmup_batches=1, dollars_per_hour=2.0)
    assert result.warmup_images == 1 and result.images == 2 and len(predictor.calls) == 3
    assert result.images_per_second > 0 and result.cost_per_100k is not None
    path = bench.save(result, "stub", root=tmp_path)
    assert json.loads(path.read_text().splitlines()[-1])["label"] == "stub"


def test_bench_needs_something_to_time(items):
    with pytest.raises(ValueError, match="nothing to time"):
        bench.bench(StubPredictor(), items, batch_size=3, warmup_batches=1)


def test_find_final_csv_wants_exactly_one(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "final.csv").write_text("image_id\n")
    assert final_run.find_final_csv(tmp_path).name == "final.csv"
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "final.csv").write_text("image_id\n")
    with pytest.raises(FileNotFoundError):
        final_run.find_final_csv(tmp_path)


def test_decrypt_failure_never_shows_the_passphrase(tmp_path, monkeypatch):
    secret = "s3cret-phrase"

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 2, stdout="", stderr=f"ERROR: Wrong password : {secret}")

    monkeypatch.setattr(final_run.subprocess, "run", fake_run)
    with pytest.raises(final_run.DecryptError) as error:
        final_run.decrypt(tmp_path / "x.bin", tmp_path / "out", secret, seven_zip="7z")
    assert secret not in str(error.value) and "***" in str(error.value)
    with pytest.raises(final_run.DecryptError, match="empty"):
        final_run.decrypt(tmp_path / "x.bin", tmp_path / "out", "", seven_zip="7z")


def test_missing_seven_zip_says_how_to_install(monkeypatch):
    monkeypatch.setattr(final_run.shutil, "which", lambda name: None)
    monkeypatch.setattr(final_run, "WINDOWS_SEVEN_ZIP", final_run.Path("/nonexistent/7z.exe"))
    with pytest.raises(final_run.DecryptError, match="Install"):
        final_run.find_seven_zip()
