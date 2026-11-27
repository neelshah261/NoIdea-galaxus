"""B7: the Saturday run. Passphrase → decrypt → runner → merge into the best dev file → validate → the command.

When and how the final images are released, and what the final file must hold, is in
[challenge_description.md, "Timeline" and "Submission file"](../../../challenge_description.md#submission-file).
The final images are for inference only: this script never trains or tunes.

    # rehearsal on dev (the Saturday checkpoint in PLAN.md), timed:
    python -m galaxus.serve.final_run --dry-run --predictor <package.module:name> --exp EXP-0001

    # Saturday, after the passphrase is published:
    python -m galaxus.serve.final_run --predictor <package.module:name> --exp EXP-00NN \
        --best-file $GALAXUS_DATA/out/submissions/<best dev file>.csv

The passphrase is asked with ``getpass``: it is never echoed, written to a file or logged. It reaches 7-Zip as a
``-p`` argument, so it is briefly visible in the local process list; run this on a machine only the team uses.
If the script stops halfway, start it again: the decrypted files are reused and the runner resumes.
"""

from __future__ import annotations

import argparse
import getpass
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ENCRYPTED_NAME = "final_encrypted.bin"
FINAL_DIR = "final"
FINAL_CSV = "final.csv"
SEVEN_ZIP_NAMES = ("7zz", "7z", "7za")
WINDOWS_SEVEN_ZIP = Path(r"C:\Program Files\7-Zip\7z.exe")


class DecryptError(RuntimeError):
    """7-Zip is missing or could not extract the archive. The message never contains the passphrase."""


def find_seven_zip() -> str:
    """The 7-Zip executable (``7zz`` on a Mac, ``7z`` elsewhere)."""
    for name in SEVEN_ZIP_NAMES:
        found = shutil.which(name)
        if found:
            return found
    if WINDOWS_SEVEN_ZIP.exists():
        return str(WINDOWS_SEVEN_ZIP)
    raise DecryptError(
        "7-Zip not found. Install it before Saturday: macOS `brew install sevenzip` (gives 7zz), "
        "Ubuntu `apt install 7zip`, Windows `winget install 7zip.7zip`."
    )


def decrypt(archive: Path, destination: Path, passphrase: str, seven_zip: str | None = None) -> None:
    """Extract ``archive`` into ``destination`` with ``passphrase``."""
    if not passphrase:
        raise DecryptError("empty passphrase")
    executable = seven_zip or find_seven_zip()
    destination.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [executable, "x", str(archive), f"-o{destination}", f"-p{passphrase}", "-y"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).replace(passphrase, "***").strip().splitlines()[-3:]
        raise DecryptError(f"7-Zip exit code {result.returncode}: {' | '.join(detail)}. Wrong passphrase?")


def find_final_csv(directory: Path) -> Path:
    """The ``final.csv`` inside the decrypted folder, wherever the archive put it."""
    candidates = sorted(directory.rglob(FINAL_CSV))
    if len(candidates) == 1:
        return candidates[0]
    found = sorted(str(path.relative_to(directory)) for path in directory.rglob("*.csv"))
    raise FileNotFoundError(f"expected one {FINAL_CSV} under {directory}, found csv files: {found[:10]}")


def final_items(passphrase_prompt: Any = getpass.getpass) -> list[Any]:
    """Decrypt the final images if needed (asking for the passphrase) and return their items."""
    from galaxus.data.load import read_part
    from galaxus.data.paths import data_dir, data_file
    from galaxus.serve.items import items_from_frame

    destination = data_dir() / FINAL_DIR
    try:
        csv = find_final_csv(destination)
        print(f"[final] reusing the decrypted files in {destination}")
    except FileNotFoundError:
        archive = data_file(ENCRYPTED_NAME)
        passphrase = passphrase_prompt("Passphrase of the final images (not echoed, not stored): ").strip()
        started = time.perf_counter()
        decrypt(archive, destination, passphrase)
        del passphrase
        print(f"[final] decrypted into {destination} in {time.perf_counter() - started:.0f} s")
        csv = find_final_csv(destination)
    return items_from_frame(read_part(csv))


def main(argv: list[str] | None = None, passphrase_prompt: Any = getpass.getpass) -> int:
    """``python -m galaxus.serve.final_run``: the whole Saturday sequence, or its rehearsal with ``--dry-run``."""
    from galaxus.serve.items import load_predictor
    from galaxus.serve.runner import items_for_part, predictions_path, read_labels, run
    from galaxus.serve.submit import git_revision, submit_command, write_file

    parser = argparse.ArgumentParser(prog="python -m galaxus.serve.final_run", description=__doc__.split("\n")[0])
    parser.add_argument("--predictor", required=True, help="package.module:name of the frozen method")
    parser.add_argument("--exp", required=True, help="experiment id of the frozen method")
    parser.add_argument("--best-file", help="validated dev submission whose final rows get replaced")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=1, help="batches in flight (thread-safe predictors only)")
    parser.add_argument("--dry-run", action="store_true", help="rehearse on the dev images; no passphrase")
    parser.add_argument("--limit", type=int, default=None, help="dry run only: the first N dev images")
    arguments = parser.parse_args(argv)
    if arguments.limit is not None and not arguments.dry_run:
        parser.error("--limit is only for --dry-run: the final file needs every final image")

    started = time.perf_counter()
    revision = git_revision()
    print(f"[final] commit {revision}, predictor {arguments.predictor}, exp {arguments.exp}")
    if arguments.dry_run:
        part, items = "dryrun-dev", items_for_part("dev", arguments.limit)
    else:
        part, items = "final", final_items(passphrase_prompt)
    if not arguments.best_file:
        print("[final] warning: no --best-file; the other rows keep sample_submission.csv's values", file=sys.stderr)

    predictor = load_predictor(arguments.predictor)
    report = run(predictor, items, arguments.exp, part, arguments.batch_size, workers=arguments.workers)
    if not report.complete:
        print(f"[final] {report.already_done + report.predicted}/{report.requested} predicted; start again to resume")
        return 1

    labels = read_labels(predictions_path(arguments.exp, part))
    wanted = {str(item.image_id) for item in items}
    name = f"{arguments.exp}_{part}.csv"
    path = write_file({image_id: labels[image_id] for image_id in wanted}, name, base=arguments.best_file)
    total = time.perf_counter() - started

    speed = report.images_per_second
    print(f"[final] valid: {path}")
    print(f"[final] {len(items)} images; this call took {total:.0f} s in total; predicting ran at {speed} img/s")
    if speed:
        print(f"[final] at that speed, {len(items)} images need {len(items) / speed / 60:.1f} min of prediction")
    print("\nA human runs this (agents never upload, AGENTS.md product rule 7):\n")
    print("  " + submit_command(path, f"{arguments.exp} {part} commit {revision}"))
    if not arguments.dry_run:
        print("\nThen, on Kaggle, SELECT this submission for the final ranking (challenge_description.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
