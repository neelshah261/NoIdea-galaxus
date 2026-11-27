"""Serve-lane benchmark grids, on top of the measurement primitives in :mod:`galaxus.serve.bench`.

``src/galaxus/serve/bench.py`` answers "how fast was this predictor?" (timing after warm-up, peak memory, the
PLAN.md "Speed" cost formula). This directory answers "what do we measure, in which order, on which hardware, and
which commands does a human run?" — the parts that are a plan rather than a library.

Nothing here starts a GPU job: AGENTS.md "Galaxus product rules" 7 forbids an agent to start a paid GPU job, so the
GPU cells of the grid are *printed as commands* and stay empty until a human runs them.

Run it from the repository root, where ``bench`` is importable:

    uv run python -m bench.speed_matrix grid
    uv run python -m bench.speed_matrix plan
    uv run python -m bench.speed_matrix fake --images 96
"""
