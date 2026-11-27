"""Saturday pre-flight: GO / NO-GO right before the final run, in one command, without spending anything.

The drill (#32, ``final_run --drill``) proves the path works on dev. This answers the questions that only matter on
the day, at the moment of running it:

1. **Submissions left today** on Kaggle (read-only ``submission-limits``; nothing is uploaded).
2. **Will it finish in time?** Final images / measured images per second, plus a buffer, against the upload-by time.
3. **The archive**: ``final_encrypted.bin`` is there and its SHA-256 matches the Kaggle data page.
4. **7-Zip** is installed (reuses ``final_run.find_seven_zip``).
5. **The best dev file** exists and passes the data lane's validator.
6. **Disk space** for the decrypted final images.

The times and the checksum come from their single sources: challenge_description.md "Submission file" (upload by
14:15 CET, deadline 14:30 CET) and the Kaggle data page (SHA-256). Override them with flags if a source changes.

    python -m galaxus.serve.preflight --best-file $GALAXUS_DATA/out/submissions/<best dev file>.csv \
        --images-per-second <from the drill or bench>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

#: challenge_description.md "Submission file": upload by about 14:15 CET on 28 Nov 2026 (13:15 UTC).
UPLOAD_BY = datetime(2026, 11, 28, 13, 15, tzinfo=timezone.utc)
#: Kaggle data page, "Final images": the SHA-256 of final_encrypted.bin.
FINAL_SHA256 = "4b37a260c9fcc0325e8ee43ec5db1519dea7ff6ced0d918fb691c1c563506465"
COMPETITION = "hackapertus-galaxus"
ENCRYPTED_NAME = "final_encrypted.bin"
#: Extra minutes for decrypting, writing, validating and uploading.
BUFFER_MINUTES = 10
#: Keep at least this many submissions for the final window (PLAN.md "Timeline" says keep >= 5).
MIN_SUBMISSIONS = 2
MIN_FREE_GB = 10.0


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool | None  # None: could not be checked (shown as WARN, never as GO)
    detail: str


def check_time(
    final_images: int, images_per_second: float | None, now: datetime, upload_by: datetime = UPLOAD_BY,
    buffer_minutes: int = BUFFER_MINUTES,
) -> Check:
    """Does prediction plus the buffer end before the upload-by time?"""
    if not images_per_second or images_per_second <= 0:
        return Check("time", None, "no measured speed: pass --images-per-second from the drill or bench")
    upload_by = upload_by.astimezone(timezone.utc)
    now = now.astimezone(timezone.utc)
    minutes = final_images / images_per_second / 60
    finish = now + timedelta(minutes=minutes + buffer_minutes)
    slack = (upload_by - finish).total_seconds() / 60
    detail = (
        f"{final_images} images at {images_per_second:g} img/s = {minutes:.1f} min + {buffer_minutes} min buffer; "
        f"done {finish.strftime('%H:%M')} UTC, upload-by {upload_by.strftime('%H:%M')} UTC ({slack:+.0f} min)"
    )
    return Check("time", slack >= 0, detail)


def parse_remaining(text: str) -> int | None:
    """``Remaining today: N`` from ``kaggle competitions submission-limits``."""
    match = re.search(r"Remaining today:\s*(\d+)", text)
    return int(match.group(1)) if match else None


def check_submissions(run: Callable[[list[str]], str] | None = None, minimum: int = MIN_SUBMISSIONS) -> Check:
    """Submissions left today, read-only. Uploads stay a human's job (AGENTS.md product rule 7)."""
    command = [sys.executable, "-m", "kaggle", "competitions", "submission-limits", COMPETITION]
    try:
        output = (run or _run)(command)
    except (OSError, subprocess.SubprocessError) as error:
        return Check("submissions", None, f"could not ask Kaggle: {type(error).__name__}")
    remaining = parse_remaining(output)
    if remaining is None:
        return Check("submissions", None, "Kaggle's answer had no 'Remaining today' line (logged in? joined?)")
    return Check("submissions", remaining >= minimum, f"{remaining} left today (want at least {minimum})")


def _run(command: list[str]) -> str:
    return subprocess.run(command, capture_output=True, text=True, timeout=60, check=False).stdout


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def check_archive(path: Path, expected: str = FINAL_SHA256) -> Check:
    if not path.is_file():
        return Check("archive", False, f"{path} is missing: download the Kaggle data first")
    actual = sha256_of(path)
    return Check("archive", actual == expected, f"SHA-256 {'matches' if actual == expected else 'DIFFERS: ' + actual}")


def check_seven_zip() -> Check:
    from galaxus.serve.final_run import DecryptError, find_seven_zip

    try:
        return Check("7-zip", True, find_seven_zip())
    except DecryptError as missing:
        return Check("7-zip", False, str(missing))


def check_best_file(path: Path | None) -> Check:
    if path is None:
        return Check("best dev file", None, "no --best-file: final rows would go over sample_submission.csv")
    if not path.is_file():
        return Check("best dev file", False, f"{path} is missing")
    from galaxus.data.validate import SubmissionInvalid, validate_submission

    try:
        frame = validate_submission(path)
    except SubmissionInvalid as invalid:
        return Check("best dev file", False, f"invalid: {invalid.problems[:2]}")
    return Check("best dev file", True, f"valid, {len(frame)} rows")


def check_disk(directory: Path, minimum_gb: float = MIN_FREE_GB) -> Check:
    free = shutil.disk_usage(directory).free / 1e9
    return Check("disk", free >= minimum_gb, f"{free:.0f} GB free (want {minimum_gb:g})")


def final_image_count() -> int:
    """Final rows = sample_submission rows that are not dev rows (works before the archive is opened)."""
    from galaxus.data.load import read_dev, read_sample_submission

    return len(set(read_sample_submission()["image_id"]) - set(read_dev()["image_id"]))


def verdict(checks: Sequence[Check]) -> str:
    if any(c.ok is False for c in checks):
        return "NO-GO"
    return "GO" if all(c.ok for c in checks) else "GO WITH WARNINGS"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m galaxus.serve.preflight", description=__doc__.split("\n")[0])
    parser.add_argument("--best-file", type=Path, default=None)
    parser.add_argument("--images-per-second", type=float, default=None, help="measured by the drill or bench")
    parser.add_argument("--upload-by", default=None, help="ISO time with zone; default from challenge_description.md")
    parser.add_argument("--skip-kaggle", action="store_true", help="do not ask Kaggle (offline)")
    parser.add_argument("--skip-sha", action="store_true", help="skip hashing the 5 GB archive")
    arguments = parser.parse_args(argv)

    from galaxus.data.paths import data_dir

    root = data_dir()
    upload_by = datetime.fromisoformat(arguments.upload_by) if arguments.upload_by else UPLOAD_BY
    now = datetime.now(timezone.utc)
    checks = [
        check_time(final_image_count(), arguments.images_per_second, now, upload_by),
        Check("submissions", None, "skipped (--skip-kaggle)") if arguments.skip_kaggle else check_submissions(),
        Check("archive", None, "skipped (--skip-sha)") if arguments.skip_sha else check_archive(root / ENCRYPTED_NAME),
        check_seven_zip(),
        check_best_file(arguments.best_file),
        check_disk(root),
    ]
    for check in checks:
        mark = {True: "OK  ", False: "FAIL", None: "WARN"}[check.ok]
        print(f"[{mark}] {check.name:14} {check.detail}")
    result = verdict(checks)
    print(f"\n{result}")
    out = root / "out" / "preflight.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"at": now.isoformat(timespec="seconds"), "verdict": result,
                                 "checks": [asdict(c) for c in checks]}) + "\n")
    return 1 if result == "NO-GO" else 0


if __name__ == "__main__":
    sys.exit(main())
