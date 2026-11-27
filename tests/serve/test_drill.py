"""#32 drill mode: the Saturday sequence rehearsed end to end on synthetic dev images.

Every image, table and archive here is made inside the test. No competition data, no network, no Kaggle
(AGENTS.md "Work → PR" 5 and "Galaxus product rules" 7). The decrypt step needs 7-Zip on the machine; where it is
not installed, the tests that extract an archive skip with that reason instead of failing.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image as PillowImage
from serve_helpers import DEV_IDS, DIMENSIONS, FAKE_CLASSES, FINAL_IDS

pytest.importorskip("galaxus.data.submission", reason="needs the data lane's writer and validator (#6)")

from galaxus.data.validate import read_submission
from galaxus.serve import final_run, runner, submit

STUB = "serve_helpers:StubPredictor"


@pytest.fixture
def seven_zip():
    """The 7-Zip executable, or a clean skip saying it is missing."""
    try:
        return final_run.find_seven_zip()
    except final_run.DecryptError as missing:
        pytest.skip(f"the decrypt step cannot be rehearsed here: {missing}")


@pytest.fixture
def best_dev_file(galaxus_data):
    """A validated "best dev file" whose rows the drill replaces, as on Saturday."""
    return submit.write_file({DEV_IDS[0]: {d: FAKE_CLASSES[d][2] for d in DIMENSIONS}}, "best_dev.csv")


def practice_archive(directory: Path, seven_zip: str, passphrase: str) -> Path:
    """A synthetic passphrase-protected 7z built here: two throwaway files, nothing from the competition."""
    source = directory / "practice"
    (source / "images").mkdir(parents=True)
    (source / "practice_final.csv").write_text("image_id,image_path\np1,images/p1.png\n", encoding="utf-8")
    PillowImage.new("RGB", (8, 8), (1, 2, 3)).save(source / "images" / "p1.png")
    archive = directory / "practice.7z"
    built = subprocess.run(
        [seven_zip, "a", f"-p{passphrase}", "-mhe=on", str(archive), "practice_final.csv", "images"],
        cwd=source,
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    return archive


def step_of(report, name):
    """One step of a report by name."""
    return next(step for step in report.steps if step["name"] == name)


def test_the_drill_walks_every_saturday_step_on_synthetic_dev_images(galaxus_data, best_dev_file, capsys):
    argv = ["--drill", "--predictor", STUB, "--exp", "EXP-0032", "--best-file", str(best_dev_file)]
    exit_code = final_run.main([*argv, "--batch-size", "1", "--drill-part", "d1"])
    out = capsys.readouterr().out
    assert exit_code == 0, out

    report = json.loads((galaxus_data / "out" / "drill" / "d1" / "report.json").read_text(encoding="utf-8"))
    assert [step["name"] for step in report["steps"]] == list(final_run.DRILL_STEPS)
    assert [step["name"] for step in report["steps"] if step["status"] == "failed"] == []
    # No --archive, so only the extraction is left out; every other step really ran.
    assert report["skipped"] == ["decrypt"] and not report["ready"] and report["ok"]
    assert report["images"] == len(DEV_IDS) and report["part"] == "dev"
    assert report["floor_images_per_second"] == final_run.THROUGHPUT_FLOOR and report["meets_floor"]
    assert "kaggle competitions submit" in out and "img/s" in out


def test_the_drill_kills_the_run_halfway_and_the_resume_finishes_it_without_duplicates(galaxus_data):
    report = final_run.drill(STUB, "EXP-0032", drill_part="d2", batch_size=1, log=None)
    assert report.killed_after == 1 and report.resumed == len(DEV_IDS) - 1
    assert step_of(report, "kill")["status"] == "ok" and step_of(report, "resume")["status"] == "ok"

    lines = runner.predictions_path("EXP-0032", "d2").read_text(encoding="utf-8").splitlines()
    ids = [json.loads(line)["image_id"] for line in lines]
    assert sorted(ids) == sorted(DEV_IDS) and len(ids) == len(set(ids))


def test_the_drill_merges_into_the_best_dev_file_and_validates_the_result(galaxus_data, best_dev_file):
    report = final_run.drill(STUB, "EXP-0032", drill_part="d3", best_file=best_dev_file, batch_size=1, log=None)
    assert step_of(report, "merge")["status"] == "ok" and step_of(report, "validate")["status"] == "ok"

    frame = read_submission(report.submission_file).set_index("image_id")
    assert frame.index.tolist() == DEV_IDS + FINAL_IDS
    # The drilled dev rows now hold the drill's answer; the rows it did not touch keep the best file's values.
    assert (frame.loc[DEV_IDS, "subject"] == FAKE_CLASSES["subject"][1]).all()
    assert frame.loc[FINAL_IDS[0], "subject"] == FAKE_CLASSES["subject"][0]
    assert report.submit_command.startswith("kaggle competitions submit -c hackapertus-galaxus -f")


def test_the_drill_reports_wall_time_speed_and_the_margin_over_the_floor(galaxus_data):
    report = final_run.drill(STUB, "EXP-0032", drill_part="d4", batch_size=1, floor=0.5, log=None)
    assert report.wall_seconds >= report.prediction_seconds > 0
    assert report.images_per_second > 0.5 and report.meets_floor
    assert report.margin_images_per_second == pytest.approx(report.images_per_second - 0.5)
    assert report.margin_factor == pytest.approx(report.images_per_second / 0.5, rel=1e-3)
    assert report.minutes_for_these_images == round(report.images / report.images_per_second / 60, 2)


def test_a_speed_under_the_floor_is_reported_and_exits_non_zero(galaxus_data, capsys):
    argv = ["--drill", "--predictor", STUB, "--exp", "E", "--drill-part", "d5", "--batch-size", "1", "--floor", "1e9"]
    assert final_run.main(argv) == 1
    out = capsys.readouterr().out
    assert "is BELOW" in out and "fall back" in out
    report = json.loads((galaxus_data / "out" / "drill" / "d5" / "report.json").read_text(encoding="utf-8"))
    # Nothing broke: the method is simply too slow, which is what the hard checkpoint in PLAN.md is there to catch.
    assert report["ok"] and not report["meets_floor"] and not report["ready"]
    assert report["margin_images_per_second"] < 0


def test_the_report_lands_in_galaxus_data_out_drill_and_never_in_the_repository(galaxus_data):
    report = final_run.drill(STUB, "EXP-0032", drill_part="d6", batch_size=1, log=None)
    path = final_run.drill_dir(report.drill_part) / final_run.DRILL_REPORT
    assert path.is_file()
    assert path.relative_to(galaxus_data) == Path("out") / "drill" / "d6" / "report.json"
    repository = Path(final_run.__file__).resolve().parents[3]
    assert repository not in path.parents and repository not in Path(report.submission_file).resolve().parents


def test_the_drill_really_decrypts_a_synthetic_passphrase_protected_archive(galaxus_data, seven_zip, tmp_path):
    passphrase = "drill-only passphrase"
    archive = practice_archive(tmp_path, seven_zip, passphrase)
    report = final_run.drill(
        STUB,
        "EXP-0032",
        drill_part="d7",
        batch_size=1,
        archive=archive,
        passphrase_prompt=lambda prompt: passphrase,
        log=None,
    )
    step = step_of(report, "decrypt")
    assert step["status"] == "ok" and "2 file(s) extracted" in step["detail"]
    assert (final_run.drill_dir("d7") / final_run.DRILL_DECRYPTED / "practice_final.csv").is_file()
    assert report.ready and report.ok and not report.skipped


def test_a_wrong_passphrase_stops_the_drill_without_showing_the_passphrase(galaxus_data, seven_zip, tmp_path):
    archive = practice_archive(tmp_path, seven_zip, "the right one")
    wrong = "not-the-passphrase"
    report = final_run.drill(
        STUB,
        "EXP-0032",
        drill_part="d8",
        batch_size=1,
        archive=archive,
        passphrase_prompt=lambda prompt: wrong,
        log=None,
    )
    step = step_of(report, "decrypt")
    assert step["status"] == "failed" and wrong not in step["detail"] and "Wrong passphrase?" in step["detail"]
    assert report.failed == ["decrypt"] and not report.ok
    # The steps after a broken decrypt are not pretended to have run.
    assert [step["name"] for step in report.steps] == ["decrypt"]
    assert report.submission_file is None and report.submit_command is None


def test_the_decrypt_step_is_wired_to_7_zip_and_asks_only_when_there_is_an_archive(galaxus_data, monkeypatch, tmp_path):
    """The same wiring as the two tests above, on a machine without 7-Zip: the executable and ``decrypt`` are faked."""
    archive = tmp_path / "practice.7z"
    archive.write_bytes(b"not really an archive: galaxus.serve.final_run.decrypt is faked here")
    monkeypatch.setattr(final_run, "find_seven_zip", lambda: "/fake/7z")

    def fake_decrypt(source, destination, passphrase, seven_zip=None):
        assert (source, passphrase, seven_zip) == (archive, "practice passphrase", "/fake/7z")
        (destination / "images").mkdir(parents=True)
        (destination / "practice_final.csv").write_text("image_id\np1\n", encoding="utf-8")
        PillowImage.new("RGB", (8, 8)).save(destination / "images" / "p1.png")

    monkeypatch.setattr(final_run, "decrypt", fake_decrypt)
    asked = []
    with_archive = final_run.drill(
        STUB,
        "EXP-0032",
        drill_part="d10",
        batch_size=1,
        archive=archive,
        passphrase_prompt=lambda prompt: asked.append(prompt) or "practice passphrase",
        log=None,
    )
    assert step_of(with_archive, "decrypt")["status"] == "ok" and with_archive.ready
    assert "2 file(s) extracted" in step_of(with_archive, "decrypt")["detail"]
    assert len(asked) == 1 and "not echoed, not stored" in asked[0]

    def never(prompt):
        pytest.fail(f"the drill asked for a passphrase without an archive: {prompt}")

    without = final_run.drill(STUB, "EXP-0032", drill_part="d11", batch_size=1, passphrase_prompt=never, log=None)
    step = step_of(without, "decrypt")
    assert step["status"] == "skipped" and "no --archive" in step["detail"] and without.ok


def test_the_drill_refuses_to_be_a_dry_run_as_well(galaxus_data):
    with pytest.raises(SystemExit):
        final_run.main(["--drill", "--dry-run", "--predictor", STUB, "--exp", "E"])
    with pytest.raises(ValueError, match="batch_size"):
        final_run.drill(STUB, "E", drill_part="d12", batch_size=0, log=None)


def test_one_batch_cannot_be_killed_halfway_and_says_so(galaxus_data):
    report = final_run.drill(STUB, "EXP-0032", drill_part="d9", batch_size=len(DEV_IDS) + 1, log=None)
    step = step_of(report, "kill")
    assert step["status"] == "skipped" and "single batch" in step["detail"]
    assert report.ok and not report.ready and report.resumed == len(DEV_IDS)
