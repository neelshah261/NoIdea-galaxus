"""Apertus v1.5 behind one interface, with three backends (issue #13, BASE_ARCHITECTURE B6).

| backend | where Apertus runs | module |
| --- | --- | --- |
| ``api`` | the Hack Apertus gateway, set up by ``LLM_*`` | :mod:`galaxus.model.apertus.api` |
| ``vllm`` | a vLLM server running the swiss-ai image | :mod:`galaxus.model.apertus.vllm` |
| ``local`` | this process, transformers on cuda, mps or cpu | :mod:`galaxus.model.apertus.local` |

    from galaxus.model.apertus import backend
    apertus = backend("api")
    answer = apertus.generate([{"role": "user", "content": "..."}], [jpeg_bytes], json_schema)

Only Apertus v1.5 runs at runtime (AGENTS.md product rule 4). Check a backend with
``python -m galaxus.model.apertus.smoke``.
"""

from __future__ import annotations

from typing import Any

from galaxus.model.apertus.backend import APERTUS_8B, APERTUS_70B, Backend, BackendError, Generation

BACKENDS = ("api", "vllm", "local")

__all__ = ["APERTUS_8B", "APERTUS_70B", "BACKENDS", "Backend", "BackendError", "Generation", "backend"]


def backend(name: str, **options: Any) -> Backend:
    """The backend called ``name``; ``options`` go to its constructor (``api``: to ``GatewayBackend.from_env``)."""
    if name == "api":
        from galaxus.model.apertus.api import GatewayBackend

        return GatewayBackend.from_env(**options)
    if name == "vllm":
        from galaxus.model.apertus.vllm import VllmBackend

        return VllmBackend(**options)
    if name == "local":
        from galaxus.model.apertus.local import LocalBackend

        return LocalBackend(**options)
    raise ValueError(f"unknown backend {name!r}, known: {', '.join(BACKENDS)}")
