"""``vllm``: Apertus served by vLLM from the swiss-ai image (PLAN.md "Speed"), through its OpenAI-compatible API.

The server runs as its own process or container, so its torch and transformers never meet ours: the reference
notebook gives vLLM its own environment for the same reason. This module is only the client. Whether that server
enforces a JSON schema is Q-0002; the smoke script answers it.

TODO(model): write down the image tag and the serve command once B6 has run on the Hack Apertus GPU. Starting a GPU
job is a human's call (AGENTS.md product rule 7); this module never starts one.
"""

from __future__ import annotations

from typing import Any

from galaxus.model.apertus.backend import APERTUS_8B
from galaxus.model.apertus.chat_http import ChatCompletionsBackend

#: vLLM's own default address for its OpenAI-compatible server.
DEFAULT_BASE_URL = "http://localhost:8000/v1"


class VllmBackend(ChatCompletionsBackend):
    """A vLLM server. Without an API key the request carries no ``Authorization`` header."""

    name = "vllm"

    def __init__(
        self, base_url: str = DEFAULT_BASE_URL, model: str = APERTUS_8B, api_key: str | None = None, **options: Any
    ) -> None:
        super().__init__(base_url, model, api_key, **options)
