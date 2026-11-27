"""B5: predictions → a validated submission file → the exact Kaggle command, printed for a human.

AGENTS.md "Galaxus product rules" 7: agents never upload or select a submission. This module stops at the command.
The file rules (sample_submission order, constant final rows before the passphrase, lowercase ``none``) live in
[challenge_description.md, "Submission file"](../../../challenge_description.md#submission-file); the data lane's
``write_submission`` applies them and validates the result.

First Kaggle file (issue #15), once the majority predictor (#12) and the writer (#6) are on the branch:

    python -m galaxus.serve.submit --exp EXP-0001 --part dev --run galaxus.model.baselines:MajorityPredictor

Without ``--run`` it only turns an existing ``predictions/<exp>/<part>.jsonl`` into a file.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

#: The competition slug, from the Kaggle links in challenge_description.md.
COMPETITION = "hackapertus-galaxus"
SUBMISSIONS_DIR = "submissions"


def git_revision() -> str:
    """The short commit of this checkout, with ``-dirty`` when there are uncommitted changes; ``unknown`` outside git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=False
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("-dirty" if dirty.stdout.strip() else "")


def submit_command(path: str | os.PathLike[str], message: str) -> str:
    """The ``kaggle competitions submit`` command a human runs for ``path``."""
    return f"kaggle competitions submit -c {COMPETITION} -f {shlex.quote(str(path))} -m {shlex.quote(message)}"


def collect_labels(
    exp: str, parts: Sequence[str], root: str | os.PathLike[str] | None = None
) -> dict[str, dict[str, str]]:
    """The labels of ``predictions/<exp>/<part>.jsonl`` for every part, merged. An image in two parts is an error."""
    from galaxus.serve.runner import predictions_path, read_labels

    merged: dict[str, dict[str, str]] = {}
    for part in parts:
        path = predictions_path(exp, part, root)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist: run galaxus.serve.runner for {exp}/{part} first")
        labels = read_labels(path)
        overlap = sorted(set(labels) & set(merged))
        if overlap:
            raise ValueError(f"{len(overlap)} image id(s) are in more than one part, first: {overlap[:3]}")
        merged.update(labels)
    return merged


def write_file(
    labels: dict[str, dict[str, str]],
    name: str,
    base: str | os.PathLike[str] | None = None,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Write ``labels`` over ``base`` (default: sample_submission.csv) into ``<root>/submissions/<name>``, validated.

    Rows without a prediction keep ``base``'s value: the constant answers of the sample, or the rows of a best dev
    file whose final rows are being replaced.

    Raises:
        SubmissionInvalid: the file would be rejected (from the data lane's validator).
        KeyError: a predicted image id is not a row of the submission.
    """
    from galaxus.data.paths import out_dir
    from galaxus.data.submission import write_submission
    from galaxus.data.validate import read_submission

    target_root = Path(root) if root is not None else out_dir()
    sample = read_submission(base) if base is not None else None
    return write_submission(labels, path=target_root / SUBMISSIONS_DIR / name, sample=sample, validate=True)


def main(argv: list[str] | None = None) -> int:
    """``python -m galaxus.serve.submit``: optionally run a predictor, then write, validate and print the command."""
    parser = argparse.ArgumentParser(prog="python -m galaxus.serve.submit", description=__doc__.split("\n")[0])
    parser.add_argument("--exp", required=True, help="experiment id whose predictions go into the file")
    parser.add_argument("--part", action="append", choices=("dev", "final"), help="part(s) to include; default dev")
    parser.add_argument("--run", metavar="PREDICTOR", help="first run this predictor (package.module:name) on the part")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base", help="file whose other rows are kept (default: sample_submission.csv)")
    parser.add_argument("--name", help="file name under $GALAXUS_DATA/out/submissions/ (default: <exp>_<parts>.csv)")
    arguments = parser.parse_args(argv)
    parts = arguments.part or ["dev"]

    if arguments.run:
        from galaxus.serve.items import load_predictor
        from galaxus.serve.runner import items_for_part, run

        predictor = load_predictor(arguments.run)
        for part in parts:
            report = run(predictor, items_for_part(part), arguments.exp, part, arguments.batch_size)
            if not report.complete:
                print(f"{arguments.exp}/{part} is incomplete; not writing a file", file=sys.stderr)
                return 1

    labels = collect_labels(arguments.exp, parts)
    revision = git_revision()
    name = arguments.name or f"{arguments.exp}_{'+'.join(parts)}.csv"
    path = write_file(labels, name, base=arguments.base)
    message = f"{arguments.exp} {'+'.join(parts)} commit {revision}"
    print(f"valid: {path} ({len(labels)} predicted rows; the other rows keep the base file's values)")
    if revision.endswith("-dirty") or revision == "unknown":
        print("warning: uncommitted changes; commit first so the file can be traced to a commit", file=sys.stderr)
    print("\nA human runs this (agents never upload, AGENTS.md product rule 7):\n")
    print("  " + submit_command(path, message))
    print("\nThen check the public score, and select the file on Kaggle if it is the one that counts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
