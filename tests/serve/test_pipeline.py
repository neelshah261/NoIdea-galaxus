"""The CI smoke test of issue #14 and the B5/B7 paths: synthetic images → stub → runner → validated file."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("galaxus.data.submission", reason="needs the data lane's writer and validator (#6)")

from serve_helpers import DEV_IDS, DIMENSIONS, FAKE_CLASSES, FINAL_IDS, StubPredictor

from galaxus.data.validate import SubmissionInvalid, read_submission
from galaxus.serve import final_run, runner, submit

STUB = "serve_helpers:StubPredictor"


def test_smoke_three_synthetic_images_to_a_validated_file(galaxus_data):
    items = runner.items_for_part("dev")
    assert [item.image_id for item in items] == DEV_IDS
    assert items[0].context == {"product_title": "p0"}

    report = runner.run(StubPredictor(), items, "EXP-0000", "dev", log=None)
    assert report.complete
    labels = submit.collect_labels("EXP-0000", ["dev"])
    path = submit.write_file(labels, "smoke.csv")

    frame = read_submission(path)
    assert path.parent == galaxus_data / "out" / "submissions"
    assert frame["image_id"].tolist() == DEV_IDS + FINAL_IDS
    dev = frame[frame.image_id.isin(DEV_IDS)]
    final = frame[frame.image_id.isin(FINAL_IDS)]
    assert (dev["subject"] == FAKE_CLASSES["subject"][1]).all()
    assert (final["subject"] == FAKE_CLASSES["subject"][0]).all()


def test_an_unknown_class_never_reaches_a_file(galaxus_data):
    with pytest.raises(SubmissionInvalid):
        submit.write_file({DEV_IDS[0]: {d: "not_a_class" for d in DIMENSIONS}}, "bad.csv")


def test_submit_cli_runs_the_predictor_and_prints_the_command_without_uploading(galaxus_data, capsys):
    assert submit.main(["--exp", "EXP-0001", "--run", STUB]) == 0
    out = capsys.readouterr().out
    assert "kaggle competitions submit -c hackapertus-galaxus -f" in out
    assert "EXP-0001_dev.csv" in out
    assert (galaxus_data / "out" / "submissions" / "EXP-0001_dev.csv").exists()


def test_missing_predictions_are_a_clear_error(galaxus_data):
    with pytest.raises(FileNotFoundError, match="run galaxus.serve.runner"):
        submit.collect_labels("EXP-9999", ["dev"])


def test_dry_run_ends_in_a_validated_timed_file(galaxus_data, capsys):
    assert final_run.main(["--dry-run", "--predictor", STUB, "--exp", "EXP-0001"]) == 0
    out = capsys.readouterr().out
    assert "valid:" in out and "img/s" in out and "kaggle competitions submit" in out
    assert (galaxus_data / "out" / "submissions" / "EXP-0001_dryrun-dev.csv").exists()


def test_final_run_decrypts_once_merges_into_the_best_file_and_never_prints_the_passphrase(
    galaxus_data, monkeypatch, capsys
):
    (galaxus_data / "final_encrypted.bin").write_bytes(b"synthetic")
    best = submit.write_file({DEV_IDS[0]: {d: FAKE_CLASSES[d][2] for d in DIMENSIONS}}, "best_dev.csv")
    secret = "correct horse battery"
    calls = []

    def fake_decrypt(archive, destination, passphrase, seven_zip=None):
        calls.append(passphrase)
        (destination / "nested" / "images").mkdir(parents=True)
        rows = [{"image_id": i, "image_path": f"images/{i}.jpg", "product_title": "t"} for i in FINAL_IDS]
        pd.DataFrame(rows).to_csv(destination / "nested" / "final.csv", index=False)

    monkeypatch.setattr(final_run, "decrypt", fake_decrypt)
    argv = ["--predictor", STUB, "--exp", "EXP-0002", "--best-file", str(best)]
    assert final_run.main(argv, passphrase_prompt=lambda prompt: secret) == 0
    assert final_run.main(argv, passphrase_prompt=lambda prompt: pytest.fail("asked twice")) == 0
    assert calls == [secret]

    out = capsys.readouterr()
    assert secret not in out.out and secret not in out.err
    assert "SELECT this submission" in out.out
    frame = read_submission(galaxus_data / "out" / "submissions" / "EXP-0002_final.csv").set_index("image_id")
    assert frame.loc[DEV_IDS[0], "subject"] == FAKE_CLASSES["subject"][2]
    assert frame.loc[DEV_IDS[1], "subject"] == FAKE_CLASSES["subject"][0]
    assert (frame.loc[FINAL_IDS, "subject"] == FAKE_CLASSES["subject"][1]).all()


def test_limit_is_refused_outside_a_dry_run(galaxus_data):
    with pytest.raises(SystemExit):
        final_run.main(["--predictor", STUB, "--exp", "E", "--limit", "5"])
