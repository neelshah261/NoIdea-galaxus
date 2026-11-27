"""``api``: Apertus through the Hack Apertus inference gateway, set up by ``LLM_*`` (README.md "Environment").

    LLM_NAME      name and version of the model
    LLM_BASE_URL  endpoint base URL
    LLM_API_KEY   your API key; LLM_API_KEY_2, LLM_API_KEY_3, ... are used in turn when one runs out (#38)

They come from the environment or, where it has none, from ``.env`` in the repository root, which git ignores
(.gitignore). Whether the gateway takes images at all is Q-0001; the smoke script answers it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from galaxus.keys import KeyRing, load_keys
from galaxus.model.apertus.backend import DEFAULT_MAX_TOKENS, BackendError, Generation, ImageInput, Message
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


def gateway_keys(environment: Mapping[str, str] | None = None, env_file: Path = ENV_FILE) -> KeyRing:
    """Every ``LLM_API_KEY*`` from ``.env`` and the environment, in ``galaxus.keys`` order; the environment wins."""
    environment = os.environ if environment is None else environment
    from_environment = {
        name: value for name, value in environment.items() if name.startswith("LLM_API_KEY") and value.strip()
    }
    return KeyRing(load_keys({**read_env_file(env_file), **from_environment}))


class GatewayBackend(ChatCompletionsBackend):
    """The Hack Apertus gateway, one API key at a time.

    With a :class:`galaxus.keys.KeyRing` (the one :meth:`from_env` builds), a request answered with HTTP 401 or 402 is
    sent again with the next key, and later requests stay on that key (#38). A busy server (429, 5xx) is retried on
    the same key instead. Keys are never printed, only their position.
    """

    name = "api"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        keys: KeyRing | None = None,
        **options: Any,
    ) -> None:
        super().__init__(base_url, model, api_key, **options)
        self.keys = keys

    def __repr__(self) -> str:
        count = len(self.keys.keys) if self.keys is not None else int(bool(self._api_key))
        return f"{type(self).__name__}(model={self.model!r}, schema_mode={self.schema_mode!r}, keys={count})"

    def generate(
        self,
        messages: Sequence[Message],
        images: Sequence[ImageInput] = (),
        json_schema: Mapping[str, Any] | None = None,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> Generation:
        if self.keys is None:
            return super().generate(messages, images, json_schema, max_tokens=max_tokens, temperature=temperature)
        return self.keys.call(
            lambda key: self._generate(
                messages, images, json_schema, max_tokens=max_tokens, temperature=temperature, api_key=key
            )
        )

    @classmethod
    def from_env(
        cls, environment: Mapping[str, str] | None = None, env_file: Path = ENV_FILE, **options: Any
    ) -> GatewayBackend:
        """The gateway as ``LLM_*`` describes it, with every key in a ring. ``options`` go to the base class."""
        settings = gateway_settings(environment, env_file)
        return cls(
            settings["LLM_BASE_URL"], settings["LLM_NAME"], keys=gateway_keys(environment, env_file), **options
        )
