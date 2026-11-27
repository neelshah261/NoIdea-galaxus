"""B7: the Saturday run. Passphrase → decrypt → runner → merge into the best dev file → validate → the command.

When and how the final images are released, and what the final file must hold, is in
[challenge_description.md, "Timeline" and "Submission file"](../../../challenge_description.md#submission-file).
The final images are for inference only: this script never trains or tunes.

    # rehearsal on dev (the Saturday checkpoint in PLAN.md), timed:
    python -m galaxus.serve.final_run --dry-run --predictor <package.module:name> --exp EXP-0001

    # dress rehearsal of this whole script on dev, kill and resume included (issue #32):
    python -m galaxus.serve.final_run --drill --predictor <package.module:name> --exp EXP-0001 \
        --best-file $GALAXUS_DATA/out/submissions/<best dev file>.csv --archive <practice archive>.7z

    # Saturday, after the passphrase is published:
    python -m galaxus.serve.final_run --predictor <package.module:name> --exp EXP-00NN \
        --best-file $GALAXUS_DATA/out/submissions/<best dev file>.csv

The passphrase is asked with ``getpass``: it is never echoed, written to a file or logged. It reaches 7-Zip as a
``-p`` argument, so it is briefly visible in the local process list; run this on a machine only the team uses.
If the script stops halfway, start it again: the decrypted files are reused and the runner resumes.

``--drill`` is that promise tested before the day it matters: see :func:`drill` and
[docs/saturday_run_sheet.md](../../../docs/saturday_run_sheet.md).
"""

from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ENCRYPTED_NAME = "final_encrypted.bin"
FINAL_DIR = "final"
FINAL_CSV = "final.csv"
SEVEN_ZIP_NAMES = ("7zz", "7z", "7za")
WINDOWS_SEVEN_ZIP = Path(r"C:\Program Files\7-Zip\7z.exe")

DRILL_DIR = "drill"
DRILL_REPORT = "report.json"
DRILL_DECRYPTED = "decrypted"
DRILL_STEPS = ("decrypt", "items", "kill", "resume", "merge", "validate", "submit-command")
OK, SKIPPED, FAILED = "ok", "skipped", "failed"
#: Images per second to plan for on Saturday. The number is the challenge's, not ours: the Sat 28 Nov 13:00 row of
#: [challenge_description.md, "Timeline"](../../../challenge_description.md) and PLAN.md "Hard checkpoint Sat 10:00",
#: which also says what to do below it. ``--floor`` overrides it when rehearsing on a machine that is not the one
#: running Saturday.
THROUGHPUT_FLOOR = 2.0


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


@dataclass(frozen=True)
class DrillStep:
    """One step of a drill: what it was, whether it held, how long it took and what it found.

    ``status`` is ``ok`` (rehearsed and it held), ``skipped`` (could not be rehearsed here, and why) or ``failed``
    (rehearsed and it broke). A skip never fails the drill; it does keep it from being ``ready``.
    """

    name: str
    status: str
    seconds: float
    detail: str


@dataclass(frozen=True)
class DrillReport:
    """What one drill proved, and the speed it ran at. Written to ``$GALAXUS_DATA/out/drill/``, never into git."""

    exp: str
    predictor: str
    commit: str
    part: str
    drill_part: str
    images: int
    killed_after: int
    resumed: int
    wall_seconds: float
    prediction_seconds: float
    images_per_second: float | None
    floor_images_per_second: float
    margin_images_per_second: float | None
    margin_factor: float | None
    meets_floor: bool
    minutes_for_these_images: float | None
    submission_file: str | None
    submit_command: str | None
    ok: bool
    ready: bool
    failed: list[str]
    skipped: list[str]
    steps: list[dict[str, Any]]
    finished_at: str


class _StopAfter:
    """A predictor that raises ``KeyboardInterrupt`` after ``batches`` batches: the drill's simulated kill.

    Nothing below this is faked. The wrapped predictor really predicts those batches and the runner really writes
    and fsyncs them, so the resume afterwards reads a half-finished file, exactly the state a crash leaves behind.
    """

    def __init__(self, predictor: Any, batches: int) -> None:
        self.predictor = predictor
        self.batches = batches
        self.seen = 0

    def predict(self, items: Any) -> Any:
        if self.seen >= self.batches:
            raise KeyboardInterrupt("drill: simulated kill halfway")
        self.seen += 1
        return self.predictor.predict(items)


class _StepFailed(Exception):
    """Internal: a drill step broke, so the steps after it make no sense. The report is still written."""


def drill_dir(drill_part: str, root: str | os.PathLike[str] | None = None) -> Path:
    """``<root>/drill/<drill_part>/``, created if needed; ``root`` defaults to ``$GALAXUS_DATA/out``.

    Everything one drill produces lives here: its report and any practice files it extracted. AGENTS.md "Galaxus
    product rules" 3: this is outside the repository, so no drill output can reach git.
    """
    from galaxus.serve.runner import _check_name

    if root is None:
        from galaxus.data.paths import out_dir

        root = out_dir()
    path = Path(root) / DRILL_DIR / _check_name("drill part", drill_part)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_drill(report: DrillReport, root: str | os.PathLike[str] | None = None) -> Path:
    """Write ``report`` to ``<root>/drill/<drill_part>/report.json`` and return the path."""
    path = drill_dir(report.drill_part, root) / DRILL_REPORT
    path.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    return path


def _rehearse_decrypt(destination: Path, archive: str | os.PathLike[str] | None, passphrase_prompt: Any) -> DrillStep:
    """The decrypt step: check 7-Zip and, with a practice archive, really extract it with a passphrase."""
    started = time.perf_counter()

    def step(status: str, detail: str) -> DrillStep:
        return DrillStep("decrypt", status, round(time.perf_counter() - started, 3), detail)

    try:
        executable = find_seven_zip()
    except DecryptError as missing:
        return step(SKIPPED, str(missing))
    if archive is None:
        return step(
            SKIPPED,
            f"7-Zip is installed ({executable}), but no --archive was given, so the extraction itself was not "
            "rehearsed. Build a practice archive (see docs/saturday_run_sheet.md) and pass it.",
        )
    passphrase = passphrase_prompt("Passphrase of the PRACTICE archive (not echoed, not stored): ").strip()
    try:
        decrypt(Path(archive), destination, passphrase, seven_zip=executable)
    except DecryptError as failure:
        return step(FAILED, str(failure))
    finally:
        del passphrase
    extracted = [path for path in destination.rglob("*") if path.is_file()]
    return step(OK, f"{len(extracted)} file(s) extracted into {destination} with 7-Zip ({executable})")


def drill(
    predictor_spec: str,
    exp: str,
    part: str = "dev",
    drill_part: str | None = None,
    best_file: str | os.PathLike[str] | None = None,
    batch_size: int = 16,
    limit: int | None = None,
    archive: str | os.PathLike[str] | None = None,
    passphrase_prompt: Any = getpass.getpass,
    floor: float = THROUGHPUT_FLOOR,
    root: str | os.PathLike[str] | None = None,
    log: Callable[[str], None] | None = print,
) -> DrillReport:
    """Rehearse the whole Saturday sequence on the dev images, and report what it cost (issue #32).

    The one run that cannot be retried is the final one, so every step of :func:`main` is walked here first, with
    the same code: the decrypt step, the run, a kill halfway and the resume after it, the merge into the best dev
    file, the validation, and the ``kaggle competitions submit`` command printed for a human. Nothing is uploaded
    (AGENTS.md "Galaxus product rules" 7) and the final images are never touched.

    Args:
        predictor_spec: ``package.module:name`` of the predictor to rehearse with.
        exp: experiment id the predictions and the file are filed under.
        part: which images to treat as the final ones: ``dev`` (the Saturday rehearsal) or ``holdout``.
        drill_part: name of this drill, used for its folder and its predictions file. Default: a UTC timestamp, so
            two drills never share a predictions file and the kill can always be rehearsed.
        best_file: the validated dev submission whose rows get replaced, as on Saturday.
        batch_size: batch size to rehearse with. The kill lands between batches, so at least two are needed.
        limit: only the first N images (a shorter rehearsal).
        archive: a **practice** passphrase-protected archive to really extract. Never the competition archive: its
            passphrase is published only on Saturday (challenge_description.md, "Timeline").
        floor: images per second to clear, default :data:`THROUGHPUT_FLOOR`.
        root: where output goes; defaults to ``$GALAXUS_DATA/out``.
        log: where the progress lines go, or None for a silent drill.

    Returns:
        The :class:`DrillReport`, already written to ``<root>/drill/<drill_part>/report.json``.
    """
    from galaxus.data.classes import ID_COLUMN
    from galaxus.data.validate import SubmissionInvalid, validate_submission
    from galaxus.serve.items import load_predictor
    from galaxus.serve.runner import items_for_part, predictions_path, read_labels, read_predictions, run
    from galaxus.serve.submit import git_revision, submit_command, write_file

    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    name = drill_part or f"drill-{part}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    folder = drill_dir(name, root)
    revision = git_revision()
    steps: list[DrillStep] = []
    images = killed = resumed = 0
    prediction_seconds = 0.0
    submission: Path | None = None
    command: str | None = None
    wall_started = time.perf_counter()

    def record(step: DrillStep) -> DrillStep:
        steps.append(step)
        if log:
            log(f"[drill] {step.status.upper():<7} {step.name}: {step.detail} ({step.seconds:.1f} s)")
        return step

    def timed(step_name: str, status: str, started: float, detail: str) -> DrillStep:
        return record(DrillStep(step_name, status, round(time.perf_counter() - started, 3), detail))

    if log:
        log(f"[drill] {name}: commit {revision}, predictor {predictor_spec}, exp {exp}, floor {floor} img/s")
    try:
        if record(_rehearse_decrypt(folder / DRILL_DECRYPTED, archive, passphrase_prompt)).status == FAILED:
            raise _StepFailed

        started = time.perf_counter()
        items = items_for_part(part, limit)
        if not items:
            timed("items", FAILED, started, f"no images in {part}: nothing to rehearse on")
            raise _StepFailed
        images = len(items)
        batches = math.ceil(images / batch_size)
        timed("items", OK, started, f"{images} image(s) of {part} in {batches} batch(es) of {batch_size}")

        predictions = predictions_path(exp, name, root)
        kill_after = batches // 2
        started = time.perf_counter()
        if kill_after < 1:
            timed(
                "kill",
                SKIPPED,
                started,
                f"{images} image(s) are a single batch of {batch_size}, so there is no halfway to be killed at. "
                "Lower --batch-size or raise --limit to rehearse the resume.",
            )
        else:
            try:
                run(_StopAfter(load_predictor(predictor_spec), kill_after), items, exp, name, batch_size, root, None)
            except KeyboardInterrupt:
                prediction_seconds += time.perf_counter() - started
            else:
                timed("kill", FAILED, started, "the simulated kill did not stop the run")
                raise _StepFailed
            killed = len(read_predictions(predictions, repair=False))
            if not 0 < killed < images:
                timed("kill", FAILED, started, f"the kill left {killed} of {images} prediction(s): not halfway")
                raise _StepFailed
            timed(
                "kill",
                OK,
                started,
                f"killed after {kill_after} of {batches} batch(es): {killed} of {images} prediction(s) on disk",
            )

        started = time.perf_counter()
        resume = run(load_predictor(predictor_spec), items, exp, name, batch_size, root, None)
        prediction_seconds += resume.seconds
        done = resume.already_done + resume.predicted
        if not resume.complete:
            timed("resume", FAILED, started, f"{done} of {images} predicted after the resume; it did not finish")
            raise _StepFailed
        resumed = resume.predicted
        timed(
            "resume",
            OK,
            started,
            f"restarted on the half-written file: skipped {resume.already_done}, predicted {resumed} more, "
            f"{done} of {images} in {predictions.name}",
        )

        started = time.perf_counter()
        labels = read_labels(predictions)
        wanted = [str(item.image_id) for item in items]
        missing = [image_id for image_id in wanted if image_id not in labels]
        if missing:
            timed("merge", FAILED, started, f"{len(missing)} image(s) have no prediction, first: {missing[:3]}")
            raise _StepFailed
        base = str(best_file) if best_file else "sample_submission.csv"
        try:
            submission = write_file(
                {image_id: labels[image_id] for image_id in wanted}, f"{exp}_{name}.csv", best_file, root
            )
        except (SubmissionInvalid, KeyError, OSError) as failure:
            timed("merge", FAILED, started, f"the merge over {base} did not produce a file: {failure}")
            raise _StepFailed from failure
        timed("merge", OK, started, f"{len(wanted)} drilled row(s) merged over {base} into {submission}")

        started = time.perf_counter()
        try:
            frame = validate_submission(submission)
        except SubmissionInvalid as invalid:
            timed("validate", FAILED, started, f"the merged file would be rejected: {invalid}")
            raise _StepFailed from invalid
        rows = frame.set_index(ID_COLUMN)
        wrong = [
            image_id
            for image_id in wanted
            if any(
                str(rows.at[image_id, dimension]) != value
                for dimension, value in labels[image_id].items()
                if dimension in rows.columns
            )
        ]
        if wrong:
            timed("validate", FAILED, started, f"{len(wrong)} drilled row(s) lost their prediction: {wrong[:3]}")
            raise _StepFailed
        timed(
            "validate",
            OK,
            started,
            f"re-read from disk: {len(frame)} valid row(s), the {len(wanted)} drilled one(s) hold the prediction",
        )

        started = time.perf_counter()
        command = submit_command(submission, f"{exp} {name} commit {revision} (drill, not the final file)")
        timed("submit-command", OK, started, "prepared for a human; a drill never uploads")
    except _StepFailed:
        pass

    speed = round(images / prediction_seconds, 3) if images and prediction_seconds > 0 else None
    failed = [step.name for step in steps if step.status == FAILED]
    skipped = [step.name for step in steps if step.status == SKIPPED]
    reached = {step.name for step in steps}
    not_reached = [step for step in DRILL_STEPS if step not in reached]
    meets_floor = floor <= 0 or (speed is not None and speed >= floor)
    report = DrillReport(
        exp=exp,
        predictor=predictor_spec,
        commit=revision,
        part=part,
        drill_part=name,
        images=images,
        killed_after=killed,
        resumed=resumed,
        wall_seconds=round(time.perf_counter() - wall_started, 3),
        prediction_seconds=round(prediction_seconds, 3),
        images_per_second=speed,
        floor_images_per_second=floor,
        margin_images_per_second=round(speed - floor, 3) if speed is not None else None,
        margin_factor=round(speed / floor, 3) if speed is not None and floor > 0 else None,
        meets_floor=meets_floor,
        minutes_for_these_images=round(images / speed / 60, 2) if speed else None,
        submission_file=str(submission) if submission is not None else None,
        submit_command=command,
        ok=not failed and not not_reached,
        ready=not failed and not skipped and not not_reached and meets_floor,
        failed=failed,
        skipped=skipped,
        steps=[asdict(step) for step in steps],
        finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    _log_drill(report, save_drill(report, root), log)
    return report


def _log_drill(report: DrillReport, path: Path, log: Callable[[str], None] | None) -> None:
    """The summary a human reads off the screen; the same numbers are in ``path``."""
    if log is None:
        return
    log(f"[drill] report: {path}")
    log(
        f"[drill] {report.images} image(s) of {report.part}: {report.wall_seconds:.1f} s wall time, of which "
        f"{report.prediction_seconds:.1f} s predicting at {report.images_per_second} img/s"
    )
    if report.images_per_second is not None and report.floor_images_per_second > 0:
        clears = "clears" if report.meets_floor else "is BELOW"
        log(
            f"[drill] that {clears} the {report.floor_images_per_second} img/s floor by "
            f"{report.margin_images_per_second:+} img/s ({report.margin_factor}x); "
            f"these {report.images} image(s) would need {report.minutes_for_these_images} min"
        )
    if not report.meets_floor:
        log("[drill] below the floor: fall back as PLAN.md 'Hard checkpoint' says, or use a faster machine")
    for step in report.skipped:
        log(f"[drill] not rehearsed here: {step} (its detail in the report says why)")
    for step in report.failed:
        log(f"[drill] BROKEN: {step} (its detail in the report says how)")
    log(f"[drill] ready for Saturday: {'yes' if report.ready else 'no, see the lines above'}")
    if report.submit_command:
        log("\nA human runs this, never an agent (AGENTS.md product rule 7). On Saturday the real run prints the")
        log("same line for the file that has the final rows merged into the best dev file:\n")
        log("  " + report.submit_command)


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
    parser.add_argument("--drill", action="store_true", help="dress rehearsal: every step below, kill and resume too")
    parser.add_argument("--limit", type=int, default=None, help="dry run or drill only: the first N dev images")
    parser.add_argument("--part", default="dev", choices=("dev", "holdout"), help="drill only: images to rehearse on")
    parser.add_argument("--drill-part", default=None, help="drill only: its name (default: drill-<part>-<timestamp>)")
    parser.add_argument("--archive", default=None, help="drill only: a PRACTICE 7z archive to really decrypt")
    parser.add_argument(
        "--floor",
        type=float,
        default=THROUGHPUT_FLOOR,
        help=f"drill only: images/s to clear (default {THROUGHPUT_FLOOR}, from challenge_description.md 'Timeline')",
    )
    arguments = parser.parse_args(argv)
    if arguments.drill and arguments.dry_run:
        parser.error("--drill is the fuller rehearsal; do not pass --dry-run as well")
    if arguments.limit is not None and not (arguments.dry_run or arguments.drill):
        parser.error("--limit is only for --dry-run or --drill: the final file needs every final image")

    if arguments.drill:
        report = drill(
            arguments.predictor,
            arguments.exp,
            part=arguments.part,
            drill_part=arguments.drill_part,
            best_file=arguments.best_file,
            batch_size=arguments.batch_size,
            limit=arguments.limit,
            archive=arguments.archive,
            passphrase_prompt=passphrase_prompt,
            floor=arguments.floor,
        )
        return 0 if report.ok and report.meets_floor else 1

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
