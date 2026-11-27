"""The commands a **human** runs for the GPU cells of the speed matrix (#33).

AGENTS.md "Galaxus product rules" 7: agents never start a paid GPU job. So this module writes the commands out
instead of running them, generated from :func:`bench.speed_matrix.grid` so the commands and the grid cannot drift
apart. Until a human has run a cell, its row stays empty.

Two things in here are **not** facts and are marked as such:

- the price of the GPU (``--dollars-per-hour``) - a provider's number, and spending money is an owner decision;
- the predictor class names, because ``galaxus.model.apertus.vllm`` and ``.local`` are the model lane's and do not
  exist yet (BASE_ARCHITECTURE.md B6). They are defaults you can override with ``--predictor``/``--local-predictor``.

    uv run python -m bench.speed_matrix plan --priority p0
"""

from __future__ import annotations

from collections.abc import Sequence

from .speed_matrix import (
    DEFAULT_LABEL,
    SUPPORTED,
    UNCONFIRMED,
    Cell,
    all_cells,
    grid,
)

#: The image PLAN.md "Speed" names for offline batched vLLM.
VLLM_IMAGE = "ghcr.io/swiss-ai/vllm_apertus_1.5_release"
#: The 8B checkpoint from README.md "Models". Issue #17: the hosts expect the 70B to be too slow.
MODEL = "swiss-ai/Apertus-v1.5-8B"
#: TODO(model): the real class names, once `galaxus.model.apertus.{vllm,local}` exist (BASE_ARCHITECTURE.md B6).
PROPOSED_VLLM_PREDICTOR = "galaxus.model.apertus.vllm:ApertusVllmPredictor"
PROPOSED_LOCAL_PREDICTOR = "galaxus.model.apertus.local:ApertusLocalPredictor"


def _fence(body: str, language: str = "bash") -> str:
    return f"```{language}\n{body.strip()}\n```"


def preconditions() -> str:
    """Name the hardware and the price before anything is measured."""
    body = """
# 1. The data, outside the repository (AGENTS.md product rule 3).
export GALAXUS_DATA=~/galaxus-data

# 2. Name the hardware. Issue #33 wants it named, not assumed: this string lands in every row.
export GPU_NAME="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
echo "$GPU_NAME"

# 3. The hourly price of that GPU, and where it comes from. Never invent one; an empty price leaves the
#    $/100k column empty instead of wrong (TODO(owner): read it off the provider's price page before you run).
export GPU_PRICE=            # e.g. 1.80
export GPU_PRICE_SOURCE=     # e.g. "RunPod L40S on-demand, 28 Nov 2026"

# 4. The commit the numbers belong to, for the EXP row.
git rev-parse --short HEAD
"""
    return _fence(body)


def probe() -> str:
    """Ask the image which precisions it supports, instead of assuming (issue #33, PLAN.md Q-0002)."""
    body = f"""
# Run this BEFORE any fp8 / awq-int4 / gptq-int4 cell. Nothing in this repository says the swiss-ai image
# supports them, and issue #17 only records that the hosts served bf16 and did not try quantization.
docker run --rm --gpus all {VLLM_IMAGE} python - <<'PROBE'
import vllm
print("vllm", vllm.__version__)
try:
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
    print("quantization methods:", sorted(QUANTIZATION_METHODS))
except Exception as unavailable:
    print("could not list them here:", type(unavailable).__name__, unavailable)
    print("fall back to: vllm serve --help | grep -A3 -i quantization")
PROBE
"""
    return _fence(body)


def band_command(
    cells: Sequence[Cell],
    part: str = "dev",
    limit: int = 256,
    label: str = DEFAULT_LABEL,
    predictor: str = PROPOSED_VLLM_PREDICTOR,
    model: str = MODEL,
) -> str:
    """One docker start that runs every vLLM cell of a band, one process per cell."""
    ids = " \\\n          ".join(cell.cell_id for cell in cells)
    body = f"""
docker run --rm --gpus all --shm-size 16g \\
  -e GALAXUS_DATA=/data -e HF_HOME=/data/hf \\
  -e GPU_NAME="$GPU_NAME" -e GPU_PRICE="$GPU_PRICE" -e GPU_PRICE_SOURCE="$GPU_PRICE_SOURCE" \\
  -v "$GALAXUS_DATA:/data" -v "$PWD:/repo" -w /repo \\
  {VLLM_IMAGE} bash -s <<'IN_CONTAINER'
set -eu
pip install -q -e .
for cell in {ids}; do
  python -m bench.speed_matrix run --cell "$cell" \\
    --predictor {predictor} \\
    --predictor-arg model={model} \\
    --part {part} --limit {limit} --warmup 1 \\
    --hardware "$GPU_NAME" \\
    ${{GPU_PRICE:+--dollars-per-hour "$GPU_PRICE"}} ${{GPU_PRICE:+--price-source "$GPU_PRICE_SOURCE"}} \\
    --label {label} || echo "cell $cell failed: its row stays empty, not guessed"
done
IN_CONTAINER
"""
    return _fence(body)


def local_band_command(
    cells: Sequence[Cell],
    part: str = "dev",
    limit: int = 256,
    label: str = DEFAULT_LABEL,
    predictor: str = PROPOSED_LOCAL_PREDICTOR,
    model: str = MODEL,
) -> str:
    """The transformers cells: same GPU, no container (BASE_ARCHITECTURE.md B6, the ``local`` backend)."""
    ids = " \\\n          ".join(cell.cell_id for cell in cells)
    body = f"""
uv sync --extra local        # TODO(design): the `local` extra (torch + transformers) is B0 in BASE_ARCHITECTURE.md
for cell in {ids}; do
  uv run python -m bench.speed_matrix run --cell "$cell" \\
    --predictor {predictor} \\
    --predictor-arg model={model} \\
    --part {part} --limit {limit} --warmup 1 \\
    --hardware "$GPU_NAME" \\
    ${{GPU_PRICE:+--dollars-per-hour "$GPU_PRICE"}} ${{GPU_PRICE:+--price-source "$GPU_PRICE_SOURCE"}} \\
    --label {label} || echo "cell $cell failed: its row stays empty, not guessed"
done
"""
    return _fence(body)


def report_command(label: str = DEFAULT_LABEL) -> str:
    """How to turn the raw rows into the table for the report and the deck."""
    body = f"""
# The raw rows (never in git):
ls "$GALAXUS_DATA/out/bench/{label}.jsonl"

# The table, with every cell nobody ran still empty:
uv run python -m bench.speed_matrix report --label {label}
"""
    return _fence(body)


def cell_ids_in(text: str) -> set[str]:
    """Every known cell id that appears in ``text``. Used to check the plan covers the grid."""
    return {cell_id for cell_id in all_cells() if cell_id in text}


def plan_markdown(
    cells: Sequence[Cell] | None = None,
    part: str = "dev",
    limit: int = 256,
    label: str = DEFAULT_LABEL,
    predictor: str = PROPOSED_VLLM_PREDICTOR,
    local_predictor: str = PROPOSED_LOCAL_PREDICTOR,
    model: str = MODEL,
) -> str:
    """The whole human-run plan for ``cells`` (default: the ``p0``/``p1`` bands of the grid)."""
    chosen = list(cells) if cells is not None else [cell for cell in grid() if cell.priority in ("p0", "p1")]
    vllm_cells = [cell for cell in chosen if cell.backend == "vllm"]
    local_cells = [cell for cell in chosen if cell.backend == "local"]
    unconfirmed = [cell for cell in chosen if cell.support == UNCONFIRMED]
    supported_vllm = [cell for cell in vllm_cells if cell.support == SUPPORTED]
    bands = sorted({cell.priority for cell in chosen})

    banner = (
        '**An agent must not run anything on this page.** AGENTS.md "Galaxus product rules" 7: agents never start '
        "a paid GPU job. Every command below is generated from `bench.speed_matrix.grid`, so it cannot drift from "
        "the grid; each run appends one row to `$GALAXUS_DATA/out/bench/` and nothing to git."
    )
    counted = (
        f"Bands here: {', '.join(bands) or 'none'} - "
        f"{len(vllm_cells)} vLLM cell(s), {len(local_cells)} transformers cell(s), "
        f"{len(unconfirmed)} with unconfirmed support."
    )
    timing = (
        "Timing excludes model load: the predictor is built before `bench` starts its clock, and the first "
        "`--warmup` batch is run untimed (`galaxus.serve.bench.bench`)."
    )
    gate = (
        "Paste the answer into the PR before running an fp8 or int4 cell. Until then those cells are "
        "`unconfirmed`, and an unconfirmed cell that is not run stays empty."
    )
    steps: list[tuple[str, str, str]] = [
        ("the data, the hardware, the price", preconditions(), ""),
        ("what does the image actually support?", probe(), gate),
    ]
    probe_step = len(steps)
    if supported_vllm:
        steps.append(("the vLLM cells", band_command(supported_vllm, part, limit, label, predictor, model), ""))
    if local_cells:
        steps.append(
            (
                "the transformers (`local`) cells, same GPU",
                local_band_command(local_cells, part, limit, label, local_predictor, model),
                "",
            )
        )
    if unconfirmed:
        steps.append(
            (
                f"the quantized cells, only after Step {probe_step} says yes",
                band_command(unconfirmed, part, limit, label, predictor, model),
                "",
            )
        )
    steps.append(("read the matrix", report_command(label), ""))

    parts = ["# Speed matrix: the GPU cells, for a human to run (#33)", "", banner, "", counted, "", timing, ""]
    for number, (title, command, after) in enumerate(steps, start=1):
        parts += [f"## Step {number} - {title}", "", command, ""]
        if after:
            parts += [after, ""]
    missing = [
        (
            f"- TODO(model): `{predictor}` and `{local_predictor}` do not exist yet (BASE_ARCHITECTURE.md B6). The "
            f"grid needs a constructor that takes `model`, `image_area`, `precision` and `prefix_caching`; "
            f"otherwise `speed_matrix run` refuses rather than mislabel a row."
        ),
        "- TODO(owner): the hourly price and its source. No price means an empty `$/100k` column, never a guess.",
        (
            "- TODO(design): the `local` extra (torch, transformers) and the `gpu` extra, listed as B0 in "
            'BASE_ARCHITECTURE.md. A worker never adds a dependency (AGENTS.md "Lanes").'
        ),
        (
            "- Cells beyond these bands: regenerate with `--priority p3` "
            "(`uv run python -m bench.speed_matrix plan --priority p3`)."
        ),
    ]
    parts += ["## What is still missing before these run", "", *missing, ""]
    return "\n".join(parts)
