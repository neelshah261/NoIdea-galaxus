"""#33: the speed matrix. One table for "Speed and resource usage" (20% of the assessment): what we run, how fast,
on what hardware and at what cost per 100,000 images.

Agents never start paid GPU jobs (AGENTS.md product rule 7). So this script does three things:

- ``plan``: write the exact commands a human runs on the GPU, grouped so each vLLM server is started once and every
  benchmark cell for it runs before it is stopped (no idle GPU minutes). Saved to ``$GALAXUS_DATA/out/bench/``.
- ``fake``: run the same cells against a stand-in predictor on this machine, to prove the pipeline and the table end
  to end without a GPU. Those rows are labelled ``fake`` and never go into the report.
- ``table``: read the benchmark results (``galaxus.serve.bench`` writes them) and print the matrix as Markdown, with
  the hardware named on each row. Raw results stay in ``$GALAXUS_DATA/out/bench/``, never in git.

Image area defaults to **512 only**: Swarup's condition is 512 px for every real run. Other areas can be added with
``--areas`` for the insights table, never for the submitted method.

    python bench/speed_matrix.py plan --predictor <package.module:name> --dollars-per-hour <GPU price>
    python bench/speed_matrix.py table
"""

from __future__ import annotations

import argparse
import itertools
import json
import shlex
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LABEL_PREFIX = "matrix"
BACKENDS = ("vllm", "local")
PRECISIONS = ("bf16", "fp8")


@dataclass(frozen=True)
class Cell:
    """One configuration to time."""

    area: int
    batch: int
    workers: int
    backend: str
    precision: str
    prefix_caching: bool

    @property
    def label(self) -> str:
        cache = "pc" if self.prefix_caching else "nopc"
        return (
            f"{LABEL_PREFIX}-{self.backend}-{self.precision}-{self.area}px-b{self.batch}-w{self.workers}-{cache}"
        )

    @property
    def server(self) -> tuple[int, str, bool] | None:
        """What a vLLM server must be started with; None for the in-process backend."""
        return (self.area, self.precision, self.prefix_caching) if self.backend == "vllm" else None


def grid(
    areas: Sequence[int],
    batches: Sequence[int],
    workers: Sequence[int],
    backends: Sequence[str],
    precisions: Sequence[str],
    prefix_caching: Sequence[bool],
) -> list[Cell]:
    """Every meaningful cell. The local backend has no server: no prefix caching switch, one request at a time,
    and bf16 only until #43 measures a quantised load."""
    cells = []
    for area, batch, worker, backend, precision, cache in itertools.product(
        areas, batches, workers, backends, precisions, prefix_caching
    ):
        if backend not in BACKENDS or precision not in PRECISIONS:
            raise ValueError(f"unknown backend {backend!r} or precision {precision!r}")
        if area < 32 or area % 16:
            raise ValueError(f"area {area} must be a multiple of 16 px (Apertus' token size)")
        if backend == "local" and (cache or worker != 1 or precision != "bf16"):
            continue
        cells.append(Cell(area, batch, worker, backend, precision, cache))
    return sorted(set(cells), key=lambda c: (c.backend, c.server or (0, "", False), c.batch, c.workers))


def server_command(cell: Cell) -> str:
    """Start the vLLM server for this cell's (area, precision, prefix caching), on the GPU machine."""
    env = {
        "MAX_PIXELS": str(cell.area * cell.area),
        "QUANT": "fp8" if cell.precision == "fp8" else "none",
        "PREFIX_CACHING": "on" if cell.prefix_caching else "off",
    }
    return " ".join(f"{k}={v}" for k, v in env.items()) + " bash bench/vllm_apertus.sh"


def bench_command(cell: Cell, predictor: str, dollars_per_hour: float | None, limit: int, part: str = "dev") -> str:
    """Time one cell from the machine with the data (``galaxus.serve.bench``)."""
    parts = [
        "python -m galaxus.serve.bench",
        f"--predictor {shlex.quote(predictor)}",
        f"--part {part}",
        f"--limit {limit}",
        f"--batch-size {cell.batch}",
        f"--workers {cell.workers}",
        f"--label {cell.label}",
    ]
    if dollars_per_hour is not None:
        parts.append(f"--dollars-per-hour {dollars_per_hour}")
    return " ".join(parts)


def plan(cells: Sequence[Cell], predictor: str, dollars_per_hour: float | None, limit: int) -> str:
    """The run sheet: one block per server start, local cells last, and a reminder to stop the GPU."""
    lines = [
        "# Speed matrix run sheet (#33)",
        "",
        "A human runs this on the GPU (AGENTS.md product rule 7). Start each server, wait for it to answer, run its",
        "benchmark lines from the machine with the data, then stop it before the next block. Stop the GPU at the end.",
        "",
        f"Predictor: `{predictor}`. Images per cell: {limit} (first batch untimed). Cells: {len(cells)}.",
        "",
    ]
    vllm = [c for c in cells if c.server]
    for server, grouped in itertools.groupby(vllm, key=lambda c: c.server):
        group = list(grouped)
        area, precision, cache = server
        lines += [
            f"## vLLM server: {area} px, {precision}, prefix caching {'on' if cache else 'off'}",
            "",
            "```bash",
            server_command(group[0]),
            "```",
            "",
            "```bash",
            *[bench_command(c, predictor, dollars_per_hour, limit) for c in group],
            "```",
            "",
        ]
    local = [c for c in cells if not c.server]
    if local:
        lines += ["## In-process (transformers, one image per forward pass)", "", "```bash"]
        lines += [bench_command(c, predictor, dollars_per_hour, limit) for c in local]
        lines += ["```", ""]
    lines += ["Then: stop the GPU, and run `python bench/speed_matrix.py table`.", ""]
    return "\n".join(lines)


def read_results(directory: Path, include_fake: bool = False) -> list[dict[str, Any]]:
    """The latest result per label from ``<directory>/*.jsonl`` written by ``galaxus.serve.bench``."""
    latest: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            label = str(row.get("label", ""))
            fake = label.startswith("fake-")
            if not (label.startswith(f"{LABEL_PREFIX}-") or (include_fake and fake)):
                continue
            if label not in latest or str(row.get("finished_at", "")) >= str(latest[label].get("finished_at", "")):
                latest[label] = row
    return [latest[label] for label in sorted(latest)]


def _shown(value: Any, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def table(rows: Iterable[dict[str, Any]]) -> str:
    """The matrix as Markdown, fastest first. Every row names its hardware; numbers are the benchmark's own."""
    rows = sorted(rows, key=lambda r: -(r.get("images_per_second") or 0))
    header = (
        "| Cell | img/s | Peak GB | $ / 100k | Hours / 100k | Device | Images timed |\n"
        "| --- | --- | --- | --- | --- | --- | --- |"
    )
    body = [
        f"| `{r['label']}` | {_shown(r.get('images_per_second'))} | {_shown(r.get('peak_memory_gb'))} | "
        f"{_shown(r.get('cost_per_100k'))} | {_shown(r.get('hours_per_100k'))} | {r.get('device', '-')} | "
        f"{r.get('images', '-')} |"
        for r in rows
    ]
    return "\n".join([header, *body]) if body else "No matrix results yet: run the plan's commands first."


class FakePredictor:
    """Stands in for Apertus on this machine: waits in proportion to the image tokens of the cell's area."""

    def __init__(self, area: int, seconds_per_kilotoken: float = 0.002) -> None:
        self.delay = seconds_per_kilotoken * (area / 16) ** 2 / 1000

    def predict(self, items: Sequence[Any]) -> list[dict[str, Any]]:
        time.sleep(self.delay * len(items))
        return [{"image_id": item.image_id, "labels": {"subject": "x"}, "confidence": None} for item in items]


def run_fake(cells: Sequence[Cell], out: Path, images: int = 64) -> Path:
    """Time every cell against :class:`FakePredictor`; rows are labelled ``fake-...`` and never reported."""
    from galaxus.serve import bench

    @dataclass(frozen=True)
    class _Item:
        image_id: str
        path: str = ""
        context: None = None

    items = [_Item(f"fake_{i:05d}") for i in range(images)]
    path = Path()
    for cell in cells:
        result = bench.bench(FakePredictor(cell.area), items, cell.batch, 1, None, cell.workers)
        path = bench.save(result, "fake-" + cell.label.removeprefix(f"{LABEL_PREFIX}-"), root=out)
    return path


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python bench/speed_matrix.py", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "fake"):
        command = sub.add_parser(name)
        command.add_argument("--areas", default="512", help="comma list; 512 only for real runs (Swarup's rule)")
        command.add_argument("--batches", default="8,16,32")
        command.add_argument("--workers", default="1,8")
        command.add_argument("--backends", default="vllm,local")
        command.add_argument("--precisions", default="bf16,fp8")
        command.add_argument("--prefix-caching", default="on,off")
        if name == "plan":
            command.add_argument("--predictor", required=True)
            command.add_argument("--dollars-per-hour", type=float, default=None)
            command.add_argument("--limit", type=int, default=256)
    show = sub.add_parser("table")
    show.add_argument("--include-fake", action="store_true")
    arguments = parser.parse_args(argv)

    from galaxus.data.paths import out_dir

    out = out_dir()
    bench_dir = out / "bench"
    if arguments.command == "table":
        print(table(read_results(bench_dir, arguments.include_fake)))
        return 0
    cells = grid(
        _ints(arguments.areas),
        _ints(arguments.batches),
        _ints(arguments.workers),
        [b.strip() for b in arguments.backends.split(",")],
        [p.strip() for p in arguments.precisions.split(",")],
        [c.strip() == "on" for c in arguments.prefix_caching.split(",")],
    )
    if arguments.command == "fake":
        run_fake(cells, out)
        print(table(read_results(bench_dir, include_fake=True)))
        return 0
    sheet = plan(cells, arguments.predictor, arguments.dollars_per_hour, arguments.limit)
    bench_dir.mkdir(parents=True, exist_ok=True)
    target = bench_dir / "matrix_plan.md"
    target.write_text(sheet, encoding="utf-8")
    print(sheet)
    print(f"saved to {target}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
