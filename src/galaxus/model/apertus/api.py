"""``api``: Apertus through the Hack Apertus inference gateway, set up by ``LLM_*`` (README.md "Environment").

    LLM_NAME      name and version of the model
    LLM_BASE_URL  endpoint base URL
    LLM_API_KEY   your API key

They come from the environment or, where it has none, from ``.env`` in the repository root, which git ignores
(.gitignore). Whether the gateway takes images at all is Q-0001; the smoke script answers it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from galaxus.model.apertus.backend import BackendError
from galaxus.model.apertus.chat_http import ChatCompletionsBackend

#: ``.env`` in the repository root, from this file's place in ``src/galaxus/model/apertus/``.
ENV_FILE = Path(__file__).resolve().parents[4] / ".env"
SETTINGS = ("LLM_NAME", "LLM_BASE_URL", "LLM_API_KEY")


def read_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    """The ``KEY=VALUE`` lines of a ``.env`` file. Comments, blank lines, ``export`` and quotes are allowed."""
    if not path.is_file():
        return {}
    values = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        key, separator, value = line.partition("=")
        if not separator or line.startswith("#"):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def gateway_settings(environment: Mapping[str, str] | None = None, env_file: Path = ENV_FILE) -> dict[str, str]:
    """``LLM_*`` from the environment, else from ``.env``.

    Raises:
        BackendError: naming every setting that is missing, never showing a value.
    """
    environment = os.environ if environment is None else environment
    from_file = read_env_file(env_file)
    settings = {key: environment.get(key) or from_file.get(key) or "" for key in SETTINGS}
    missing = [key for key, value in settings.items() if not value]
    if missing:
        raise BackendError(
            f"api: {', '.join(missing)} not set. Put them in the environment or in .env in the repository root "
            "(git ignores it); README.md 'Environment' names them."
        )
    return settings


class GatewayBackend(ChatCompletionsBackend):
    """The Hack Apertus gateway."""

    name = "api"

    @classmethod
    def from_env(
        cls, environment: Mapping[str, str] | None = None, env_file: Path = ENV_FILE, **options: Any
    ) -> GatewayBackend:
        """The gateway as ``LLM_*`` describes it. ``options`` go to :class:`ChatCompletionsBackend`."""
        settings = gateway_settings(environment, env_file)
        return cls(settings["LLM_BASE_URL"], settings["LLM_NAME"], settings["LLM_API_KEY"], **options)
