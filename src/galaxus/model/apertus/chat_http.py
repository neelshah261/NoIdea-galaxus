"""A small client for OpenAI-compatible chat completions: the protocol the gateway and vLLM both speak.

Standard library only (``urllib``), so neither backend needs a dependency. Images travel as base64 ``data:`` URLs
in the last user message, before its text. A JSON schema travels as ``response_format`` (OpenAI structured
outputs), as ``guided_json`` (vLLM's older parameter), or not at all; which one a server enforces is Q-0002, and
the smoke script tries them. Token logprobs are asked for with ``extra_body={"logprobs": True}`` (OpenAI's field,
which vLLM and the gateway accept) and come back in ``Generation.logprobs`` (#75).

The API key goes only into the ``Authorization`` header: never into a message, a log line or an EXP row.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from galaxus.model.apertus.backend import (
    DEFAULT_MAX_TOKENS,
    BackendError,
    Generation,
    ImageInput,
    Message,
    data_url,
    last_user_index,
)

#: ``transport(url, headers, body, timeout) -> (status, body)``. Tests pass a fake one: no network in ``ci``.
Transport = Callable[[str, Mapping[str, str], bytes, float], tuple[int, bytes]]
SCHEMA_MODES = ("response_format", "guided_json", "none")
#: Statuses worth one more try: the server is busy or briefly down.
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


def urllib_transport(url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> tuple[int, bytes]:
    """POST ``body`` to ``url``. An HTTP error status is returned, not raised; a network failure raises ``OSError``."""
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def chat_completions_url(base_url: str) -> str:
    """The chat completions endpoint under an OpenAI-style base URL (``.../v1``)."""
    url = base_url.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise ValueError("the base URL must start with http:// or https://")
    return url if url.endswith("/chat/completions") else f"{url}/chat/completions"


def _excerpt(raw: bytes, limit: int = 300) -> str:
    text = raw.decode("utf-8", errors="replace").strip()
    return text if len(text) <= limit else f"{text[:limit]}…"


def _content_text(content: Any) -> str:
    """The text of an answer, whether the server sends a string or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


class ChatCompletionsBackend:
    """Apertus behind an OpenAI-compatible ``/chat/completions`` endpoint."""

    name = "chat"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        schema_mode: str = "response_format",
        extra_body: Mapping[str, Any] | None = None,
        timeout: float = 120.0,
        retries: int = 2,
        backoff: float = 1.0,
        transport: Transport | None = None,
    ) -> None:
        if schema_mode not in SCHEMA_MODES:
            raise ValueError(f"schema_mode must be one of {', '.join(SCHEMA_MODES)}, got {schema_mode!r}")
        if not model:
            raise ValueError("a model name is needed")
        self.url = chat_completions_url(base_url)
        self.model = model
        self.schema_mode = schema_mode
        self.extra_body = dict(extra_body or {})
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self._api_key = api_key or None
        self._transport = transport or urllib_transport

    def __repr__(self) -> str:
        key = "set" if self._api_key else "none"
        return f"{type(self).__name__}(model={self.model!r}, schema_mode={self.schema_mode!r}, api_key={key})"

    def payload(
        self,
        messages: Sequence[Message],
        images: Sequence[ImageInput] = (),
        json_schema: Mapping[str, Any] | None = None,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """The request body for ``messages`` and ``images``."""
        chat = [dict(message) for message in messages]
        if images:
            index = last_user_index(chat)
            text = chat[index].get("content", "")
            if not isinstance(text, str):
                raise ValueError("the user message that takes the images must have plain-text content")
            parts: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": data_url(image)}} for image in images]
            parts.append({"type": "text", "text": text})
            chat[index]["content"] = parts
        body: dict[str, Any] = {
            "model": self.model,
            "messages": chat,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_schema is not None and self.schema_mode == "response_format":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": dict(json_schema), "strict": True},
            }
        elif json_schema is not None and self.schema_mode == "guided_json":
            body["guided_json"] = dict(json_schema)
        body.update(self.extra_body)
        return body

    def generate(
        self,
        messages: Sequence[Message],
        images: Sequence[ImageInput] = (),
        json_schema: Mapping[str, Any] | None = None,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> Generation:
        return self._generate(
            messages, images, json_schema, max_tokens=max_tokens, temperature=temperature, api_key=self._api_key
        )

    def _generate(
        self,
        messages: Sequence[Message],
        images: Sequence[ImageInput],
        json_schema: Mapping[str, Any] | None,
        *,
        max_tokens: int,
        temperature: float,
        api_key: str | None,
    ) -> Generation:
        """One request with ``api_key``; busy servers are retried on that same key."""
        body = json.dumps(
            self.payload(messages, images, json_schema, max_tokens=max_tokens, temperature=temperature)
        ).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        started = time.perf_counter()
        for attempt in range(self.retries + 1):
            try:
                status, raw = self._transport(self.url, headers, body, self.timeout)
            except OSError as error:
                if attempt == self.retries:
                    raise BackendError(f"{self.name}: no answer from the server: {error}") from error
                time.sleep(self.backoff * 2**attempt)
                continue
            if status in RETRY_STATUSES and attempt < self.retries:
                time.sleep(self.backoff * 2**attempt)
                continue
            break
        if status != 200:
            raise BackendError(f"{self.name}: HTTP {status}: {_excerpt(raw)}", status=status)
        return self._parse(raw, time.perf_counter() - started)

    def _parse(self, raw: bytes, seconds: float) -> Generation:
        try:
            response = json.loads(raw)
            choice = response["choices"][0]
            text = _content_text(choice["message"].get("content"))
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
            raise BackendError(f"{self.name}: not a chat completion: {_excerpt(raw)}") from error
        usage = response.get("usage") or {}
        # Optional, so a malformed block means no logprobs, not a failed request.
        logprobs = choice.get("logprobs")
        content = logprobs.get("content") if isinstance(logprobs, dict) else None
        return Generation(
            text=text,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
            model=response.get("model"),
            seconds=seconds,
            logprobs=content if isinstance(content, list) else None,
        )
