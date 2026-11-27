"""Saturday pre-flight: every check on synthetic inputs. No Kaggle call, no competition data, no network."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from galaxus.serve import preflight

NOON = datetime(2026, 11, 28, 12, 0, tzinfo=timezone.utc)  # 13:00 CET, the passphrase


def test_time_check_go_and_no_go():
    fast = preflight.check_time(5292, 10.0, NOON)
    assert fast.ok is True and "+" in fast.detail  # ~9 min + 10 min buffer, well before 13:15 UTC
    slow = preflight.check_time(5292, 1.0, NOON)
    assert slow.ok is False  # ~88 min: misses the upload-by time
    unknown = preflight.check_time(5292, None, NOON)
    assert unknown.ok is None


def test_submissions_parsed_from_kaggle_output_without_uploading():
    calls = []

    def fake(command):
        calls.append(command)
        return "Submissions today: 3\nLifetime submissions: 9\nRemaining today: 17\n"

    check = preflight.check_submissions(fake)
    assert check.ok is True and "17 left" in check.detail
    assert "submission-limits" in calls[0] and "submit" not in calls[0]
    assert preflight.check_submissions(lambda c: "Remaining today: 1").ok is False
    assert preflight.check_submissions(lambda c: "error: not logged in").ok is None


def test_archive_checksum(tmp_path):
    archive = tmp_path / "final_encrypted.bin"
    archive.write_bytes(b"synthetic")
    good = hashlib.sha256(b"synthetic").hexdigest()
    assert preflight.check_archive(archive, good).ok is True
    assert preflight.check_archive(archive, "0" * 64).ok is False
    assert preflight.check_archive(tmp_path / "missing.bin").ok is False


def test_verdict():
    ok = preflight.Check("a", True, "")
    warn = preflight.Check("b", None, "")
    fail = preflight.Check("c", False, "")
    assert preflight.verdict([ok, ok]) == "GO"
    assert preflight.verdict([ok, warn]) == "GO WITH WARNINGS"
    assert preflight.verdict([ok, warn, fail]) == "NO-GO"


def test_disk_and_missing_best_file(tmp_path):
    assert preflight.check_disk(tmp_path, minimum_gb=0).ok is True
    assert preflight.check_best_file(None).ok is None
    assert preflight.check_best_file(tmp_path / "nope.csv").ok is False


def test_times_in_other_zones_are_compared_and_shown_in_utc():
    cet = datetime.fromisoformat("2026-11-28T14:15+01:00")
    check = preflight.check_time(600, 10.0, NOON, upload_by=cet)
    assert check.ok is True and "upload-by 13:15 UTC" in check.detail
