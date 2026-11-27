"""#33: the speed matrix — image area x batch size x backend x precision x prefix caching.

Speed and resources are 20% of the score, so this is one table that says **what we run, how fast, on what, and at
what cost**. Three parts, matching what an agent may and may not do (AGENTS.md "Galaxus product rules" 7):

1. **The grid** (:func:`grid`): every cell of the product of the five axes, each with a support status, a reason and
   a priority. A cell that the repository does not document as supported is ``unconfirmed`` and is gated behind a
   probe command, never behind a guess.
2. **The runner** (:func:`run_cell`): one cell, timed through :func:`galaxus.serve.bench.bench`, with the cost per
   100,000 images taken from :func:`galaxus.serve.bench.cost_per_100k` (PLAN.md "Speed"; it is not re-derived here)
   and the hardware **named by the person running it**, never detected and assumed.
3. **The plan** (:mod:`bench.gpu_plan`): the exact commands a human runs for the GPU cells. Agents never start a
   paid GPU job, so those cells stay ``pending`` — empty, never filled with a plausible number.

A ``pending``, ``skipped`` or ``failed`` cell carries **no numbers at all**; :class:`CellResult` refuses to be built
with any. Raw records go to ``$GALAXUS_DATA/out/bench/<label>.jsonl`` and never into git (AGENTS.md product rule 3);
:func:`save` refuses a path inside the repository. ``results/metrics.json`` is written only by the model lane's eval
script, so the hand-over format is proposed in the PR's "For design" notes instead.

    uv run python -m bench.speed_matrix grid --format md
    uv run python -m bench.speed_matrix plan
    uv run python -m bench.speed_matrix fake --images 96
    uv run python -m bench.speed_matrix report
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from galaxus.data import images
from galaxus.serve import bench as serve_bench
from galaxus.serve import runner as serve_runner

from .fake_backend import FAKE_BACKEND, NO_WEIGHTS

#: The five axes of the grid (issue #33, PLAN.md "Speed").
#: Image area: the ablation named in PLAN.md "Speed"; the middle one is the shared default in ``galaxus.data.images``.
IMAGE_AREAS = (384, images.IMAGE_SIZE, 768)
#: Batch size: 1 is the latency floor, the middle one is the runner's default, the last one is throughput-bound.
BATCH_SIZES = (1, serve_runner.DEFAULT_BATCH_SIZE, 4 * serve_runner.DEFAULT_BATCH_SIZE)
#: The two product backends of BASE_ARCHITECTURE.md section 1. ``api`` is not here: it is someone else's hardware.
BACKENDS = ("vllm", "local")
#: bf16 is the documented main line; the rest are the quantizations issue #33 asks about, support unconfirmed.
PRECISIONS = ("bf16", "fp8", "awq-int4", "gptq-int4")
PREFIX_CACHING = (True, False)

#: The reference point the precision ablation is run at: the defaults both other lanes already use.
REFERENCE_AREA = images.IMAGE_SIZE
REFERENCE_BATCH = serve_runner.DEFAULT_BATCH_SIZE

SUPPORTED = "supported"
UNCONFIRMED = "unconfirmed"
NOT_APPLICABLE = "not-applicable"

PENDING = "pending"
DONE = "done"
FAILED = "failed"
SKIPPED = "skipped"
STATUSES = (PENDING, DONE, FAILED, SKIPPED)

#: Default file name under ``$GALAXUS_DATA/out/bench/``.
DEFAULT_LABEL = "speed_matrix"
#: Every number of a cell. A cell that was not run has all of them empty, and :class:`CellResult` enforces that.
NUMERIC_FIELDS = (
    "images",
    "warmup_images",
    "seconds",
    "images_per_second",
    "peak_memory_gb",
    "dollars_per_hour",
    "cost_per_100k",
    "hours_per_100k",
    "vision_tokens_per_image",
)
#: The cell axes a predictor must accept so that a row is really the row it claims to be.
CELL_ARGUMENTS = ("image_area", "precision", "prefix_caching")

REPO_ROOT = Path(__file__).resolve().parents[1]
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
#: Everything printed stays ASCII: this table is read in a Windows console as often as in a markdown viewer.
_EMPTY = "-"


class CellError(RuntimeError):
    """A cell cannot be run as described (bad axes, or a predictor that does not match the cell)."""


@dataclass(frozen=True)
class Hardware:
    """The hardware a cell ran on, **named by the person who ran it**.

    Issue #33 asks for the hardware named rather than assumed, so this is an input, not a detection: a GPU name that
    the benchmark guessed from the driver is recorded separately, as ``detected_device``, only to cross-check.
    ``dollars_per_hour`` is the provider's price; without it there is no cost per 100,000 images, and that cell of
    the table stays empty rather than being filled with an invented price.
    """

    name: str
    dollars_per_hour: float | None = None
    price_source: str = ""

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("name the hardware: issue #33 asks for the hardware named, not assumed")
        if self.dollars_per_hour is not None:
            if self.dollars_per_hour <= 0:
                raise ValueError(f"dollars_per_hour must be positive, got {self.dollars_per_hour}")
            if not str(self.price_source).strip():
                raise ValueError("a price needs a price_source: where it comes from, so nobody has to trust a number")


@dataclass(frozen=True)
class Cell:
    """One cell of the grid: the five axes of issue #33, and nothing measured."""

    image_area: int
    batch_size: int
    backend: str
    precision: str
    prefix_caching: bool

    def __post_init__(self) -> None:
        if not isinstance(self.image_area, int) or self.image_area % images.PATCH_SIZE or self.image_area < 1:
            raise ValueError(
                f"image_area must be a positive multiple of galaxus.data.images.PATCH_SIZE "
                f"({images.PATCH_SIZE}), got {self.image_area!r}"
            )
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size!r}")
        if self.backend not in (*BACKENDS, FAKE_BACKEND):
            raise ValueError(f"backend must be one of {(*BACKENDS, FAKE_BACKEND)}, got {self.backend!r}")
        if self.precision not in (*PRECISIONS, NO_WEIGHTS):
            raise ValueError(f"precision must be one of {(*PRECISIONS, NO_WEIGHTS)}, got {self.precision!r}")

    @property
    def cell_id(self) -> str:
        """Stable id, used on the command line and as the key of a saved record."""
        cache = "cache_on" if self.prefix_caching else "cache_off"
        return f"a{self.image_area}-b{self.batch_size}-{self.backend}-{self.precision}-{cache}"

    @property
    def at_reference_point(self) -> bool:
        """True at the area and batch size the other lanes already default to."""
        return self.image_area == REFERENCE_AREA and self.batch_size == REFERENCE_BATCH

    @property
    def support(self) -> str:
        return self._support[0]

    @property
    def support_reason(self) -> str:
        return self._support[1]

    @property
    def priority(self) -> str:
        """Run order: ``p0`` first, ``n/a`` never. Not the issue labels; this orders GPU minutes."""
        status = self.support
        if status == NOT_APPLICABLE:
            return "n/a"
        if status == UNCONFIRMED:
            return "p2" if self.at_reference_point else "p3"
        if self.backend == FAKE_BACKEND:
            return "p0"
        return "p0" if (self.backend == "vllm" and self.prefix_caching) else "p1"

    @property
    def _support(self) -> tuple[str, str]:
        """(status, reason). Only what the repository documents counts as supported."""
        if self.backend == FAKE_BACKEND:
            if self.precision != NO_WEIGHTS or self.prefix_caching:
                return NOT_APPLICABLE, f"the {FAKE_BACKEND} backend has no weights and no prompt"
            return SUPPORTED, "CPU only: proves the machinery, measures no model"
        if self.precision == NO_WEIGHTS:
            return NOT_APPLICABLE, f"{NO_WEIGHTS!r} means no weights at all; only the {FAKE_BACKEND} backend has none"
        if self.backend == "local":
            if self.prefix_caching:
                return NOT_APPLICABLE, "prefix caching is a vLLM feature; the transformers backend has no prompt cache"
            if self.precision != "bf16":
                return NOT_APPLICABLE, (
                    "a quantized transformers load would need a new dependency, and a worker never adds one "
                    "(AGENTS.md 'Lanes'); raise it as a design issue first"
                )
            return SUPPORTED, "transformers on one GPU, bf16 (BASE_ARCHITECTURE.md B6, the local backend)"
        if self.precision == "bf16":
            return SUPPORTED, "the documented main line: offline batched vLLM, bf16 (PLAN.md 'Speed')"
        return UNCONFIRMED, (
            f"nothing in the repository says the swiss-ai vLLM image supports {self.precision}: run the probe in "
            f"`python -m bench.speed_matrix plan` and record the answer before this cell is run"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "image_area": self.image_area,
            "batch_size": self.batch_size,
            "backend": self.backend,
            "precision": self.precision,
            "prefix_caching": self.prefix_caching,
            "support": self.support,
            "support_reason": self.support_reason,
            "priority": self.priority,
        }


def grid(include_not_applicable: bool = False) -> list[Cell]:
    """Every product cell of the five axes, in run order (priority first).

    ``not-applicable`` combinations (prefix caching without vLLM, a quantization that would need a new dependency)
    are part of the product but are left out unless asked for, so the table only lists cells somebody could run.
    """
    cells = [
        Cell(area, batch, backend, precision, caching)
        for area in IMAGE_AREAS
        for batch in BATCH_SIZES
        for backend in BACKENDS
        for precision in PRECISIONS
        for caching in PREFIX_CACHING
    ]
    if not include_not_applicable:
        cells = [cell for cell in cells if cell.support != NOT_APPLICABLE]
    return sorted(cells, key=_run_order)


def fake_grid() -> list[Cell]:
    """The CPU cells an agent may run itself: the area and batch axes with no model behind them."""
    return [Cell(area, batch, FAKE_BACKEND, NO_WEIGHTS, False) for area in IMAGE_AREAS for batch in BATCH_SIZES]


def _run_order(cell: Cell) -> tuple[Any, ...]:
    return (
        cell.priority,
        BACKENDS.index(cell.backend) if cell.backend in BACKENDS else len(BACKENDS),
        PRECISIONS.index(cell.precision) if cell.precision in PRECISIONS else len(PRECISIONS),
        not cell.prefix_caching,
        cell.image_area,
        cell.batch_size,
    )


def all_cells() -> dict[str, Cell]:
    """Every cell this module knows, by id: the whole product plus the CPU cells."""
    return {cell.cell_id: cell for cell in [*grid(include_not_applicable=True), *fake_grid()]}


def cell_from_id(cell_id: str) -> Cell:
    """The cell with this id.

    Raises:
        CellError: no cell has that id, with the nearest ids to look at.
    """
    cells = all_cells()
    if cell_id in cells:
        return cells[cell_id]
    near = [known for known in cells if known.startswith(cell_id[:8])][:5]
    raise CellError(f"no cell {cell_id!r}; ids look like {next(iter(cells))!r}. Nearest: {near or 'none'}")


@dataclass(frozen=True)
class CellResult:
    """One row of the matrix. Measured, or visibly empty — never in between.

    A row that is not ``done`` carries no number and no hardware: issue #33 forbids filling an unrun cell with a
    plausible figure, so :meth:`__post_init__` rejects it instead of trusting the caller.
    """

    cell_id: str
    image_area: int
    batch_size: int
    backend: str
    precision: str
    prefix_caching: bool
    status: str
    support: str
    priority: str
    note: str = ""
    hardware: str | None = None
    price_source: str | None = None
    detected_device: str | None = None
    images: int | None = None
    warmup_images: int | None = None
    seconds: float | None = None
    images_per_second: float | None = None
    peak_memory_gb: float | None = None
    memory_note: str | None = None
    dollars_per_hour: float | None = None
    cost_per_100k: float | None = None
    hours_per_100k: float | None = None
    vision_tokens_per_image: float | None = None
    axes_applied: bool = False
    predictor: str | None = None
    labels_source: str | None = None
    part: str | None = None
    finished_at: str | None = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")
        if self.status != DONE:
            filled = [name for name in NUMERIC_FIELDS if getattr(self, name) is not None]
            if filled or self.hardware is not None:
                raise ValueError(
                    f"{self.cell_id} is {self.status}, so it must stay empty, but {filled or ['hardware']} are "
                    f"filled: a cell that has not been run is never given a number (#33)"
                )
            return
        missing = [name for name in ("hardware", "images", "images_per_second") if getattr(self, name) is None]
        if missing:
            raise ValueError(
                f"{self.cell_id} is done but {missing} are missing: a measured row names its hardware and its speed"
            )
        if self.cost_per_100k is not None and self.dollars_per_hour is None:
            raise ValueError(f"{self.cell_id}: a cost per 100k without an hourly price is not a measurement")

    @property
    def is_machinery_check(self) -> bool:
        """True when this row proves the pipeline rather than the product, so it never enters the speed table."""
        return self.backend == FAKE_BACKEND or (self.status == DONE and not self.axes_applied)

    def to_record(self) -> dict[str, Any]:
        """The JSON record saved under ``$GALAXUS_DATA/out/bench/``."""
        return {**asdict(self), "is_machinery_check": self.is_machinery_check}


def pending_result(cell: Cell, note: str = "") -> CellResult:
    """The empty row of a cell nobody has run. Every number stays ``None``."""
    return CellResult(
        cell_id=cell.cell_id,
        image_area=cell.image_area,
        batch_size=cell.batch_size,
        backend=cell.backend,
        precision=cell.precision,
        prefix_caching=cell.prefix_caching,
        status=PENDING,
        support=cell.support,
        priority=cell.priority,
        note=note or cell.support_reason,
    )


def skipped_result(cell: Cell, why: str) -> CellResult:
    """A cell deliberately not run (not applicable, or support unconfirmed). Also empty."""
    row = pending_result(cell, why)
    return CellResult(**{**asdict(row), "status": SKIPPED})


def failed_result(cell: Cell, why: str) -> CellResult:
    """A cell that was attempted and broke. Empty as well: a half-run cell has no speed."""
    row = pending_result(cell, why)
    return CellResult(**{**asdict(row), "status": FAILED})


def check_predictor_matches_cell(cell: Cell, predictor: Any) -> list[str]:
    """The axes of ``cell`` that ``predictor`` confirms it is configured for, or an error if it contradicts one.

    A predictor that exposes ``image_area``, ``precision``, ``prefix_caching`` or ``backend`` is checked against the
    cell, so a row cannot quietly be labelled 512² while the model was served at 768².

    Raises:
        CellError: the predictor says it is configured differently from the cell.
    """
    confirmed = []
    for name in (*CELL_ARGUMENTS, "backend"):
        actual = getattr(predictor, name, None)
        if actual is None:
            continue
        expected = getattr(cell, name)
        if actual != expected:
            raise CellError(
                f"{cell.cell_id}: the predictor reports {name}={actual!r} but the cell says {expected!r}; "
                f"fix the predictor's arguments rather than the label of the row"
            )
        confirmed.append(name)
    return confirmed


def run_cell(
    cell: Cell,
    predictor: Any,
    items: Sequence[Any],
    hardware: Hardware,
    warmup_batches: int = 1,
    axes_applied: bool = True,
    predictor_spec: str | None = None,
    part: str | None = None,
) -> CellResult:
    """Time one cell and return its row.

    The timing, the peak memory and the cost per 100,000 images all come from :func:`galaxus.serve.bench.bench`,
    which implements PLAN.md "Speed"; nothing is recomputed here.

    Args:
        axes_applied: whether the cell's axes really reached the predictor. ``False`` marks the row a machinery
            check, because a row whose axes were ignored describes no configuration.

    Raises:
        CellError: the cell is not applicable, or the predictor contradicts it.
    """
    if cell.support == NOT_APPLICABLE:
        raise CellError(f"{cell.cell_id} is not applicable: {cell.support_reason}")
    check_predictor_matches_cell(cell, predictor)
    measured = serve_bench.bench(predictor, items, cell.batch_size, warmup_batches, hardware.dollars_per_hour)
    notes = [cell.support_reason]
    if not axes_applied:
        notes.append("the cell's axes were NOT applied to the predictor: machinery check, not a configuration")
    if cell.backend in BACKENDS and measured.peak_memory_gb is None:
        notes.append(f"no GPU memory reading on this device ({measured.device}): peak memory stays empty")
    if cell.backend == FAKE_BACKEND:
        notes.append("the fake backend measures image preparation only, never Apertus")
    return CellResult(
        cell_id=cell.cell_id,
        image_area=cell.image_area,
        batch_size=cell.batch_size,
        backend=cell.backend,
        precision=cell.precision,
        prefix_caching=cell.prefix_caching,
        status=DONE,
        support=cell.support,
        priority=cell.priority,
        note="; ".join(notes),
        hardware=hardware.name,
        price_source=hardware.price_source or None,
        detected_device=measured.device,
        images=measured.images,
        warmup_images=measured.warmup_images,
        seconds=measured.seconds,
        images_per_second=measured.images_per_second,
        peak_memory_gb=measured.peak_memory_gb,
        memory_note=measured.memory_note,
        dollars_per_hour=measured.dollars_per_hour,
        cost_per_100k=measured.cost_per_100k,
        hours_per_100k=measured.hours_per_100k,
        vision_tokens_per_image=getattr(predictor, "vision_tokens_per_image", None),
        axes_applied=axes_applied,
        predictor=predictor_spec or type(predictor).__name__,
        labels_source=getattr(predictor, "labels_source", None),
        part=part,
        finished_at=measured.finished_at,
    )


def build_predictor(
    spec: str,
    cell: Cell,
    pass_cell_args: bool = True,
    extra: Mapping[str, Any] | None = None,
) -> tuple[Any, bool]:
    """``(predictor, axes_applied)`` from ``"package.module:name"`` for one cell.

    With ``pass_cell_args`` the cell's axes are handed to the constructor, so the row is the configuration it claims
    to be. A target that accepts none of them is an error naming the signature the grid needs, rather than a row
    that silently measures something else. ``extra`` holds whatever else that predictor needs (``model=...``).

    Raises:
        CellError: the target takes none of the cell arguments while they were asked for.
    """
    import importlib
    import inspect

    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"predictor must look like 'package.module:name', got {spec!r}")
    target = getattr(importlib.import_module(module_name), attribute)
    if not pass_cell_args:
        if extra:
            raise CellError("--predictor-arg needs a constructor to pass them to: drop --no-cell-args")
        from galaxus.serve.items import load_predictor

        return load_predictor(spec), False
    wanted = {"image_area": cell.image_area, "precision": cell.precision, "prefix_caching": cell.prefix_caching}
    if not (isinstance(target, type) or callable(target)):
        raise CellError(f"{spec} is not callable, so the cell's axes cannot be applied; use --no-cell-args")
    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):
        parameters = {}
    takes_everything = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    accepted = {name: value for name, value in wanted.items() if takes_everything or name in parameters}
    if not accepted:
        raise CellError(
            f"{spec} accepts none of {CELL_ARGUMENTS}, so a row from it would not describe a grid cell. "
            f"TODO(model): the grid needs a constructor like "
            f"{attribute}(model: str, image_area: int, precision: str, prefix_caching: bool). "
            f"To benchmark a predictor that ignores the axes anyway, pass --no-cell-args; its row is then recorded "
            f"as a machinery check."
        )
    predictor = target(**accepted, **dict(extra or {}))
    if not callable(getattr(predictor, "predict", None)):
        raise TypeError(f"{spec} did not give an object with a predict(items) method")
    return predictor, len(accepted) == len(wanted)


def matrix_path(label: str = DEFAULT_LABEL, root: str | Path | None = None) -> Path:
    """``<root>/bench/<label>.jsonl``; ``root`` defaults to ``$GALAXUS_DATA/out``, never a path inside the repo."""
    if not _SAFE_LABEL.match(label):
        raise ValueError(f"label {label!r} must be letters, digits, '.', '_' or '-', starting with a letter or digit")
    if root is None:
        from galaxus.data.paths import out_dir

        root = out_dir()
    return _outside_the_repo(Path(root)) / serve_bench.BENCH_DIR / f"{label}.jsonl"


def _outside_the_repo(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ValueError(
            f"{resolved} is inside the repository. Raw benchmark results go to $GALAXUS_DATA/out/bench/ and never "
            f"into git (AGENTS.md 'Galaxus product rules' 3, issue #33)"
        )
    return resolved


def save(results: Iterable[CellResult], label: str = DEFAULT_LABEL, root: str | Path | None = None) -> Path:
    """Append rows to ``<root>/bench/<label>.jsonl`` and return the file."""
    path = matrix_path(label, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(json.dumps(result.to_record(), ensure_ascii=False) + "\n")
    return path


def load(path: str | Path) -> list[dict[str, Any]]:
    """The records of a matrix file, oldest first; empty when the file does not exist."""
    file = Path(path)
    if not file.exists():
        return []
    records = []
    for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as broken:
            raise ValueError(f"{file}:{number} is not a matrix record: {broken}") from broken
    return records


def latest_by_cell(records: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """The last record of each cell id: a re-run replaces an earlier one."""
    latest: dict[str, Mapping[str, Any]] = {}
    for record in records:
        cell_id = str(record.get("cell_id", ""))
        if cell_id:
            latest[cell_id] = record
    return latest


@dataclass
class MatrixView:
    """The grid joined with whatever has been measured: product rows, machinery rows, and what is still missing."""

    product: list[dict[str, Any]] = field(default_factory=list)
    machinery: list[dict[str, Any]] = field(default_factory=list)

    @property
    def done(self) -> list[dict[str, Any]]:
        return [row for row in self.product if row["status"] == DONE]

    @property
    def pending(self) -> list[dict[str, Any]]:
        return [row for row in self.product if row["status"] != DONE]


def matrix_view(records: Iterable[Mapping[str, Any]] = (), cells: Sequence[Cell] | None = None) -> MatrixView:
    """Every cell of ``cells`` (default: the product grid) as a row, measured where a record exists."""
    latest = latest_by_cell(records)
    view = MatrixView()
    for cell in cells if cells is not None else grid():
        record = latest.get(cell.cell_id)
        row = dict(record) if record else pending_result(cell).to_record()
        row.setdefault("is_machinery_check", False)
        (view.machinery if row.get("is_machinery_check") else view.product).append(row)
    known = {row["cell_id"] for row in [*view.product, *view.machinery]}
    for cell_id, record in latest.items():
        if cell_id in known:
            continue
        target = view.machinery if record.get("is_machinery_check") else view.product
        target.append(dict(record))
    return view


def _number(value: Any, digits: int = 2) -> str:
    if value is None:
        return _EMPTY
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


HEADERS = (
    "prio",
    "area px",
    "batch",
    "backend",
    "precision",
    "prefix cache",
    "img/s",
    "peak GB",
    "$/100k",
    "vis. tokens",
    "hardware",
    "status",
)


def _row_cells(row: Mapping[str, Any]) -> list[str]:
    return [
        str(row.get("priority", _EMPTY)),
        str(row["image_area"]),
        str(row["batch_size"]),
        str(row["backend"]),
        str(row["precision"]),
        "on" if row["prefix_caching"] else "off",
        _number(row.get("images_per_second")),
        _number(row.get("peak_memory_gb"), 3),
        _number(row.get("cost_per_100k"), 2),
        _number(row.get("vision_tokens_per_image"), 0),
        str(row.get("hardware") or _EMPTY),
        str(row.get("status", PENDING)),
    ]


def _table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = ["| " + " | ".join(HEADERS) + " |", "| " + " | ".join("---" for _ in HEADERS) + " |"]
    lines += ["| " + " | ".join(_row_cells(row)) + " |" for row in rows]
    return "\n".join(lines)


def matrix_markdown(records: Iterable[Mapping[str, Any]] = (), cells: Sequence[Cell] | None = None) -> str:
    """The matrix as markdown. An empty cell has not been run; no number in it is an estimate."""
    view = matrix_view(records, cells)
    formula = (
        "`cost per 100k = $/h / (img/s * 3600) * 100000`, computed by `galaxus.serve.bench.cost_per_100k` "
        '(PLAN.md "Speed"). "area px" is the target area per side. Vision tokens follow '
        "`galaxus.data.images.PATCH_SIZE`; TODO(model): Apertus' patch size is Q-0002."
    )
    empty = (
        f"`{_EMPTY}` means **not run**. Every figure below was measured on the hardware named in its row; an empty "
        "cell is never an estimate."
    )
    machinery = (
        "CPU rows with no model behind them, or rows whose axes never reached the predictor. They prove the "
        "harness runs; they say nothing about Apertus."
    )
    parts = ["## Speed matrix (#33)", "", formula, "", empty, ""]
    if view.product:
        parts += [f"{len(view.done)} of {len(view.product)} product cells measured.", "", _table(view.product)]
    else:
        parts += ["No product cell has been measured yet: every GPU cell waits for a human (`plan`)."]
    if view.machinery:
        parts += ["", "### Machinery checks (not product numbers)", "", machinery, "", _table(view.machinery)]
    return "\n".join(parts) + "\n"


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m bench.speed_matrix", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("grid", help="print the grid definition")
    show.add_argument("--format", default="md", choices=("md", "jsonl", "ids"))
    show.add_argument("--priority", default=None, help="comma-separated, e.g. p0,p1")
    show.add_argument("--backend", default=None, choices=(*BACKENDS, FAKE_BACKEND))
    show.add_argument("--include-not-applicable", action="store_true")

    plan = sub.add_parser("plan", help="print the exact commands a human runs for the GPU cells")
    plan.add_argument("--priority", default="p0,p1", help="comma-separated bands to write commands for")
    plan.add_argument("--part", default="dev", choices=("holdout", "dev"))
    plan.add_argument("--limit", type=int, default=256)
    plan.add_argument("--label", default=DEFAULT_LABEL)
    plan.add_argument("--predictor", default=None, help="the vLLM predictor, once the model lane has one")
    plan.add_argument("--local-predictor", default=None, help="the transformers predictor")
    plan.add_argument("--model", default=None, help="the checkpoint to serve")

    run = sub.add_parser("run", help="run ONE cell (a human runs this on the GPU; an agent never does)")
    run.add_argument("--cell", required=True, help="a cell id from `grid --format ids`")
    run.add_argument("--predictor", required=True, help="package.module:name")
    run.add_argument("--predictor-arg", action="append", default=[], metavar="KEY=VALUE")
    run.add_argument("--hardware", required=True, help="the GPU as nvidia-smi names it; never guessed")
    run.add_argument("--dollars-per-hour", type=float, default=None)
    run.add_argument("--price-source", default="", help="where the price comes from (required with a price)")
    run.add_argument("--part", default="dev", choices=("holdout", "dev"))
    run.add_argument("--limit", type=int, default=256)
    run.add_argument("--warmup", type=int, default=1)
    run.add_argument("--no-cell-args", action="store_true", help="do not pass the cell axes to the predictor")
    run.add_argument("--label", default=DEFAULT_LABEL)

    fake = sub.add_parser("fake", help="run the CPU cells on synthetic images (no GPU, no competition data)")
    fake.add_argument("--images", type=int, default=96, help="synthetic images to generate")
    fake.add_argument("--images-dir", default=None, help="where to write them (default: a temporary directory)")
    fake.add_argument("--warmup", type=int, default=1)
    fake.add_argument("--label", default=DEFAULT_LABEL)
    fake.add_argument("--root", default=None, help="output root (default: $GALAXUS_DATA/out)")

    report = sub.add_parser("report", help="print the matrix, pending cells empty")
    report.add_argument("--label", default=DEFAULT_LABEL)
    report.add_argument("--root", default=None)
    report.add_argument("--from", dest="from_path", default=None, help="read this jsonl instead")

    return parser.parse_args(argv)


def _selected(arguments: argparse.Namespace) -> list[Cell]:
    cells = grid(include_not_applicable=getattr(arguments, "include_not_applicable", False))
    if getattr(arguments, "backend", None):
        cells = [cell for cell in cells if cell.backend == arguments.backend]
    if getattr(arguments, "priority", None):
        wanted = {band.strip() for band in arguments.priority.split(",") if band.strip()}
        cells = [cell for cell in cells if cell.priority in wanted]
    return cells


def _predictor_arguments(pairs: Sequence[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"--predictor-arg must look like key=value, got {pair!r}")
        text = value.strip()
        if text.lower() in ("true", "false"):
            parsed[key.strip()] = text.lower() == "true"
        else:
            try:
                parsed[key.strip()] = int(text)
            except ValueError:
                parsed[key.strip()] = text
    return parsed


def _run_one(arguments: argparse.Namespace) -> int:
    cell = cell_from_id(arguments.cell)
    hardware = Hardware(arguments.hardware, arguments.dollars_per_hour, arguments.price_source)
    extra = _predictor_arguments(arguments.predictor_arg)
    predictor, axes_applied = build_predictor(arguments.predictor, cell, not arguments.no_cell_args, extra)
    items = serve_runner.items_for_part(arguments.part, arguments.limit)
    try:
        result = run_cell(
            cell,
            predictor,
            items,
            hardware,
            arguments.warmup,
            axes_applied,
            arguments.predictor,
            arguments.part,
        )
    except (CellError, ValueError, RuntimeError) as broken:
        result = failed_result(cell, f"{type(broken).__name__}: {broken}")
        path = save([result], arguments.label)
        print(f"{cell.cell_id} failed, recorded empty in {path}: {broken}", file=sys.stderr)
        return 1
    path = save([result], arguments.label)
    print(json.dumps(result.to_record(), indent=2))
    print(f"saved to {path}")
    return 0


def _run_fake(arguments: argparse.Namespace) -> int:
    from .fake_backend import FakeBackendPredictor, synthetic_items

    directory = Path(arguments.images_dir) if arguments.images_dir else Path(tempfile.mkdtemp(prefix="galaxus-bench-"))
    items = synthetic_items(arguments.images, directory)
    hardware = Hardware(f"{serve_bench.device_and_peak_memory()[0]} (no GPU: machinery check)")
    results = []
    for cell in fake_grid():
        if len(items) <= arguments.warmup * cell.batch_size:
            results.append(skipped_result(cell, f"{len(items)} synthetic images are too few for this batch size"))
            continue
        predictor = FakeBackendPredictor(cell.image_area, cell.precision, cell.prefix_caching)
        results.append(run_cell(cell, predictor, items, hardware, arguments.warmup, True, "bench.fake_backend", None))
    path = save(results, arguments.label, arguments.root)
    print(matrix_markdown([result.to_record() for result in results], fake_grid()))
    print(f"{len(items)} synthetic images in {directory}")
    print(f"saved to {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """``python -m bench.speed_matrix``: see the module docstring for the subcommands."""
    arguments = _parse_arguments(argv)
    if arguments.command == "grid":
        cells = _selected(arguments)
        if arguments.format == "ids":
            print("\n".join(cell.cell_id for cell in cells))
        elif arguments.format == "jsonl":
            print("\n".join(json.dumps(cell.as_dict()) for cell in cells))
        else:
            print(matrix_markdown((), cells))
        return 0
    if arguments.command == "plan":
        from . import gpu_plan

        print(
            gpu_plan.plan_markdown(
                _selected(arguments),
                arguments.part,
                arguments.limit,
                arguments.label,
                arguments.predictor or gpu_plan.PROPOSED_VLLM_PREDICTOR,
                arguments.local_predictor or gpu_plan.PROPOSED_LOCAL_PREDICTOR,
                arguments.model or gpu_plan.MODEL,
            )
        )
        return 0
    if arguments.command == "run":
        return _run_one(arguments)
    if arguments.command == "fake":
        return _run_fake(arguments)
    path = Path(arguments.from_path) if arguments.from_path else matrix_path(arguments.label, arguments.root)
    print(matrix_markdown(load(path)))
    print(f"read {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
