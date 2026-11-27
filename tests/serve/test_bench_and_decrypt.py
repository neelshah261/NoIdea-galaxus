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


def _final_csv(folder, ids=("img_f1", "img_f2")):
    folder.mkdir(parents=True, exist_ok=True)
    rows = "".join(f"{image_id},images/{image_id}.jpg\n" for image_id in ids)
    (folder / "final.csv").write_text("image_id,image_path\n" + rows, encoding="utf-8")


def _never_asked(prompt):
    raise AssertionError("the passphrase must not be asked for here")


def test_no_final_csv_decrypts_once_then_reads_it(galaxus_data, monkeypatch):
    (galaxus_data / "final_encrypted.bin").write_bytes(b"x")
    decrypted = []

    def fake_decrypt(archive, destination, passphrase):
        decrypted.append(passphrase)
        _final_csv(destination / "hackapertus-final")

    monkeypatch.setattr(final_run, "decrypt", fake_decrypt)
    items = final_run.final_items(lambda prompt: " phrase ")
    assert decrypted == ["phrase"] and [item.image_id for item in items] == ["img_f1", "img_f2"]


def test_one_final_csv_is_reused_without_a_passphrase(galaxus_data, monkeypatch):
    _final_csv(galaxus_data / "final" / "deep" / "er")
    monkeypatch.setattr(final_run, "decrypt", lambda *a: pytest.fail("decrypted again"))
    assert len(final_run.final_items(_never_asked)) == 2


def test_two_final_csv_stop_naming_both_without_prompting_or_decrypting(galaxus_data, monkeypatch):
    _final_csv(galaxus_data / "final" / "a")
    _final_csv(galaxus_data / "final" / "b")
    monkeypatch.setattr(final_run, "decrypt", lambda *a: pytest.fail("decrypted again"))
    with pytest.raises(final_run.FinalFilesError, match="found 2") as stopped:
        final_run.final_items(_never_asked)
    assert "a" in str(stopped.value) and "b" in str(stopped.value) and "Nothing was decrypted" in str(stopped.value)


def test_the_saturday_command_stops_cleanly_on_two_final_csv(galaxus_data, capsys):
    _final_csv(galaxus_data / "final" / "a")
    _final_csv(galaxus_data / "final" / "b")
    code = final_run.main(["--predictor", "serve_helpers:StubPredictor", "--exp", "EXP-0123"],
                          passphrase_prompt=_never_asked)
    assert code == 1 and "[final] stopped:" in capsys.readouterr().err


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
