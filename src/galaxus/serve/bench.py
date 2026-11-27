"""B7: images per second, peak memory and the cost of 100,000 images, for section 7 of the technical report.

Cost formula from PLAN.md "Speed": ``cost per 100k = $/h ÷ (img/s × 3600) × 100,000``. The hourly price is an input
(``--dollars-per-hour``) because it depends on where the GPU runs; nothing here assumes a price.

The first ``--warmup`` batches are run but not timed (model load, CUDA graphs, caches). Nothing is written to the
predictions files: the benchmark measures the predictor alone. The end-to-end speed of a real run, writes included,
is in each ``<part>.runs.jsonl`` of :mod:`galaxus.serve.runner`.

    python -m galaxus.serve.bench --predictor galaxus.model.baselines:MajorityPredictor --part dev --limit 256 \
        --batch-size 16 --dollars-per-hour <price of the GPU>
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BENCH_DIR = "bench"
SECONDS_PER_HOUR = 3600
PER_IMAGES = 100_000


@dataclass(frozen=True)
class BenchResult:
    """One benchmark: the numbers section 7 of the technical report asks for."""

    images: int
    warmup_images: int
    batch_size: int
    seconds: float
    images_per_second: float
    device: str
    peak_memory_gb: float | None
    memory_note: str
    dollars_per_hour: float | None
    cost_per_100k: float | None
    hours_per_100k: float
    finished_at: str


def cost_per_100k(dollars_per_hour: float, images_per_second: float) -> float:
    """Dollars for 100,000 images at this speed and hourly price (PLAN.md "Speed")."""
    if images_per_second <= 0:
        raise ValueError("images_per_second must be positive")
    return dollars_per_hour / (images_per_second * SECONDS_PER_HOUR) * PER_IMAGES


def hours_per_100k(images_per_second: float) -> float:
    """Hours one device needs for 100,000 images."""
    return PER_IMAGES / images_per_second / SECONDS_PER_HOUR


def _torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _reset_peak_memory() -> None:
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def device_and_peak_memory() -> tuple[str, float | None, str]:
    """(device name, peak memory in GB or None, what the memory number means)."""
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e9
        return torch.cuda.get_device_name(), round(peak, 3), "torch.cuda.max_memory_allocated in this process"
    mps = getattr(getattr(torch, "backends", None), "mps", None) if torch is not None else None
    if mps is not None and mps.is_available():
        allocated = torch.mps.driver_allocated_memory() / 1e9
        return "Apple MPS", round(allocated, 3), "torch.mps.driver_allocated_memory at the end (current, not peak)"
    return f"CPU ({platform.processor() or platform.machine()})", None, "no GPU memory: CPU run, or torch not installed"


def bench(
    predictor: Any,
    items: Sequence[Any],
    batch_size: int = 16,
    warmup_batches: int = 1,
    dollars_per_hour: float | None = None,
) -> BenchResult:
    """Time ``predictor`` over ``items`` after ``warmup_batches`` untimed batches."""
    if batch_size < 1 or warmup_batches < 0:
        raise ValueError("batch_size must be at least 1 and warmup_batches at least 0")
    warmup = list(items[: warmup_batches * batch_size])
    timed = list(items[warmup_batches * batch_size :])
    if not timed:
        raise ValueError(f"{len(items)} items leave nothing to time after {warmup_batches} warm-up batch(es)")
    for start in range(0, len(warmup), batch_size):
        predictor.predict(warmup[start : start + batch_size])
    _reset_peak_memory()
    started = time.perf_counter()
    for start in range(0, len(timed), batch_size):
        predictor.predict(timed[start : start + batch_size])
    seconds = time.perf_counter() - started
    speed = len(timed) / seconds if seconds > 0 else float("inf")
    device, peak, note = device_and_peak_memory()
    return BenchResult(
        images=len(timed),
        warmup_images=len(warmup),
        batch_size=batch_size,
        seconds=round(seconds, 3),
        images_per_second=round(speed, 3),
        device=device,
        peak_memory_gb=peak,
        memory_note=note,
        dollars_per_hour=dollars_per_hour,
        cost_per_100k=round(cost_per_100k(dollars_per_hour, speed), 4) if dollars_per_hour is not None else None,
        hours_per_100k=round(hours_per_100k(speed), 4),
        finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def save(result: BenchResult, label: str, root: Path | None = None) -> Path:
    """Append ``result`` to ``<root>/bench/<label>.jsonl`` (default root: ``$GALAXUS_DATA/out``)."""
    from galaxus.serve.runner import _check_name

    if root is None:
        from galaxus.data.paths import out_dir

        root = out_dir()
    path = Path(root) / BENCH_DIR / f"{_check_name('label', label)}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"label": label, **asdict(result)}) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    """``python -m galaxus.serve.bench``: time a predictor on the first N images of a part."""
    from galaxus.serve.items import load_predictor
    from galaxus.serve.runner import items_for_part

    parser = argparse.ArgumentParser(prog="python -m galaxus.serve.bench", description=__doc__.split("\n")[0])
    parser.add_argument("--predictor", required=True)
    parser.add_argument("--part", default="dev", choices=("holdout", "dev"))
    parser.add_argument("--limit", type=int, default=256, help="images to use, warm-up included")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1, help="untimed batches first")
    parser.add_argument("--dollars-per-hour", type=float, default=None, help="GPU price, for the cost per 100k")
    parser.add_argument("--label", default=None, help="name of the result file (default: the predictor name)")
    arguments = parser.parse_args(argv)

    predictor = load_predictor(arguments.predictor)
    items = items_for_part(arguments.part, arguments.limit)
    result = bench(predictor, items, arguments.batch_size, arguments.warmup, arguments.dollars_per_hour)
    label = arguments.label or arguments.predictor.rsplit(":", 1)[-1]
    path = save(result, label)
    print(json.dumps(asdict(result), indent=2))
    print(f"saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
