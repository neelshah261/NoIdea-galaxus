"""One command for the whole paid GPU session, so the GPU never sits idle (plan #40: low cost, 512 px always).

Run it on the GPU machine, in a checkout that has the heads code (#41, branch ``model/heads``) next to the serve lane,
with ``$GALAXUS_DATA`` holding the Kaggle files and ``$HF_TOKEN`` set in the shell (never on the command line).

Steps, in order; every one resumes, so a re-run after a crash continues where it stopped:

1. check: data folder, Hugging Face token present (not printed), a CUDA GPU.
2. cache: the data lane's 512 px image cache for train and dev (CPU, minutes).
3. extract-train, 4. extract-dev: Apertus 8B, one forward pass per image (the only GPU steps).
5. fit: heads + calibration on CPU, EXP row on the holdout, heads file saved.
6. dev-file: dev predictions with the heads, validated submission file, the Kaggle command printed (not run).
7. label-errors: the bonus csv from out-of-fold predictions (never committed; a human sends it).

After step 4 the GPU is no longer needed: with ``--gpu-only`` the session stops there, so the CPU steps can run on
a laptop after copying ``$GALAXUS_DATA/cache/features/``. The summary at the end gives minutes per step, the
measured images per second of the Apertus pass, and the estimated cost. **Stop the GPU right after.**

    python bench/gpu_session.py --dry-run
    python bench/gpu_session.py --dollars-per-hour <price of this GPU> --exp-id EXP-00NN
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

PREDICTOR = "galaxus.model.heads:predictor_from_env"
GPU_STEPS = ("extract-train", "extract-dev")


@dataclass
class Step:
    name: str
    command: list[str]
    gpu: bool = False
    images: int | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class StepResult:
    name: str
    seconds: float
    returncode: int
    images: int | None = None

    @property
    def images_per_second(self) -> float | None:
        return round(self.images / self.seconds, 3) if self.images and self.seconds > 0 else None


def python(*args: str) -> list[str]:
    return [sys.executable, *args]


def plan(exp_id: str, heads_file: str, train_images: int | None, dev_images: int | None) -> list[Step]:
    """The session's steps. ``heads_file`` is where ``heads fit`` saves the heads (see :func:`heads_path`)."""
    cache = (
        "from galaxus.data.images import fill_image_cache; from galaxus.data.load import read_train, read_dev; "
        "[print(fill_image_cache(list(zip(f.image_id, f.image_file)))) for f in (read_train(), read_dev())]"
    )
    return [
        Step("cache", python("-c", cache)),
        Step("extract-train", python("-m", "galaxus.model.heads", "extract", "--part", "train"), True, train_images),
        Step("extract-dev", python("-m", "galaxus.model.heads", "extract", "--part", "dev"), True, dev_images),
        Step("fit", python("-m", "galaxus.model.heads", "fit", "--exp-id", exp_id)),
        Step(
            "dev-file",
            python("-m", "galaxus.serve.submit", "--exp", exp_id, "--run", PREDICTOR),
            env={"HEADS_MODEL": heads_file},
        ),
        Step("label-errors", python("-m", "galaxus.model.heads", "label-errors")),
    ]


def heads_path() -> str:
    """Where ``python -m galaxus.model.heads fit`` saves the heads for the default model, layers and pool."""
    import hashlib

    from galaxus.data.paths import out_dir

    try:
        from galaxus.model import heads
    except ImportError:
        return "$GALAXUS_DATA/out/heads/heads-<key>.npz"

    key = heads.feature_key(
        os.environ.get(heads.APERTUS_ENV, "swiss-ai/Apertus-v1.5-8B"), heads.default_layers(), heads.DEFAULT_POOL
    )
    return str(out_dir() / "heads" / f"heads-{hashlib.sha256(key.encode()).hexdigest()[:8]}.npz")


def preconditions(environ: Callable[[str], str | None] = os.environ.get) -> list[str]:
    """What is missing before any paid minute is spent. The token is only checked for presence."""
    problems = []
    if not environ("GALAXUS_DATA"):
        problems.append("$GALAXUS_DATA is not set")
    if not environ("HF_TOKEN"):
        problems.append("$HF_TOKEN is not set (the Apertus models are gated; set it in the shell, never in git)")
    if importlib.util.find_spec("galaxus.model.heads") is None:
        problems.append("galaxus.model.heads is missing: use a checkout with branch model/heads merged in")
    if shutil.which("nvidia-smi") is None:
        problems.append("no nvidia-smi: this does not look like a CUDA GPU machine")
    return problems


def run_session(
    steps: Sequence[Step],
    start: str | None = None,
    gpu_only: bool = False,
    execute: Callable[[Step], int] | None = None,
    log: Callable[[str], None] = print,
) -> list[StepResult]:
    """Run the steps from ``start`` on; stop at the first failure, or after the GPU steps with ``gpu_only``."""
    names = [s.name for s in steps]
    if start is not None and start not in names:
        raise ValueError(f"unknown step {start!r}, steps: {', '.join(names)}")
    run = execute or _execute
    results: list[StepResult] = []
    for step in steps[names.index(start) if start else 0 :]:
        log(f"[session] {step.name}{' (GPU)' if step.gpu else ''}: {' '.join(step.command[1:])[:120]}")
        began = time.perf_counter()
        code = run(step)
        result = StepResult(step.name, round(time.perf_counter() - began, 1), code, step.images)
        results.append(result)
        if code != 0:
            log(f"[session] {step.name} failed (exit {code}); fix it, then re-run with --from {step.name}")
            break
        if gpu_only and step.name == GPU_STEPS[-1]:
            log("[session] GPU steps done (--gpu-only). STOP THE GPU, then copy cache/features/ and run --from fit")
            break
    return results


def _execute(step: Step) -> int:
    return subprocess.run(step.command, env={**os.environ, **step.env}, check=False).returncode


def summary(results: Sequence[StepResult], dollars_per_hour: float | None) -> dict[str, object]:
    total = sum(r.seconds for r in results)
    gpu = [r for r in results if r.name in GPU_STEPS]
    images = sum(r.images or 0 for r in gpu)
    gpu_seconds = sum(r.seconds for r in gpu)
    return {
        "steps": [{**asdict(r), "images_per_second": r.images_per_second} for r in results],
        "session_minutes": round(total / 60, 1),
        "apertus_images_per_second": round(images / gpu_seconds, 3) if images and gpu_seconds else None,
        "estimated_cost_usd": round(dollars_per_hour * total / 3600, 2) if dollars_per_hour else None,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _counts() -> tuple[int | None, int | None]:
    try:
        from galaxus.data.load import read_dev, read_train

        return len(read_train()), len(read_dev())
    except Exception:  # noqa: BLE001 - only for the summary's images per second; the steps report real errors
        return None, None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python bench/gpu_session.py", description=__doc__.split("\n")[0])
    parser.add_argument("--exp-id", default="EXP-0010", help="EXP id for the heads row")
    parser.add_argument("--dollars-per-hour", type=float, default=None)
    parser.add_argument("--from", dest="start", default=None, help="resume from this step")
    parser.add_argument("--gpu-only", action="store_true", help="stop after the GPU steps")
    parser.add_argument("--dry-run", action="store_true", help="print the steps and checks; run nothing")
    arguments = parser.parse_args(argv)

    train, dev = _counts() if os.environ.get("GALAXUS_DATA") else (None, None)
    head_file = heads_path() if os.environ.get("GALAXUS_DATA") else "$GALAXUS_DATA/out/heads/heads-<key>.npz"
    steps = plan(arguments.exp_id, head_file, train, dev)
    problems = preconditions()
    if arguments.dry_run:
        for step in steps:
            print(f"{step.name:14}{' GPU' if step.gpu else '    '}  {' '.join(step.command[1:])}")
        print("\nmissing: " + "; ".join(problems) if problems else "\nall preconditions met")
        return 0
    if problems:
        print("not starting, nothing was spent:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        return 1
    results = run_session(steps, arguments.start, arguments.gpu_only)
    report = summary(results, arguments.dollars_per_hour)
    from galaxus.data.paths import out_dir

    path = out_dir() / "gpu_session.jsonl"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report) + "\n")
    print(json.dumps(report, indent=2))
    print("\nSTOP THE GPU NOW: idle time is billed too.")
    return 0 if all(r.returncode == 0 for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
