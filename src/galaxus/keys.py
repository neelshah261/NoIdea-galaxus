"""Use one API key at a time and switch to the next when its credits run out.

Keys: LLM_API_KEY first, then LLM_API_KEY_2, LLM_API_KEY_3, ... (see .env.example). Run with
`uv run --env-file .env ...` so they are in the environment.

    ring = KeyRing()
    reply = ring.call(lambda key: GatewayBackend(base_url, model, key).generate(...))

Keys are never printed, only their position.
"""

import os
import re
import sys
import threading

# 402 Payment Required: the key's credits are used up. 401 Unauthorized: the key is invalid or revoked.
SWITCH_ON = {401, 402}


def load_keys(env=os.environ) -> list[str]:
    numbered = sorted(
        (int(m.group(1)), value.strip())
        for name, value in env.items()
        if (m := re.fullmatch(r"LLM_API_KEY_(\d+)", name)) and value.strip()
    )
    first = [env["LLM_API_KEY"].strip()] if env.get("LLM_API_KEY", "").strip() else []
    return list(dict.fromkeys(first + [value for _, value in numbered]))  # drop duplicates, keep order


def status_of(exc: BaseException) -> int | None:
    """HTTP status of an error: galaxus BackendError.status, openai/httpx/requests status_code, urllib code."""
    for obj in (exc, getattr(exc, "response", None)):
        for attr in ("status_code", "status", "code"):
            code = getattr(obj, attr, None)
            if isinstance(code, int):
                return code
    return None


class KeyRing:
    def __init__(self, keys: list[str] | None = None):
        self.keys = load_keys() if keys is None else list(keys)
        if not self.keys:
            raise RuntimeError("no API key: set LLM_API_KEY (and LLM_API_KEY_2, ...) in .env")
        self.index = 0
        self._lock = threading.Lock()

    @property
    def key(self) -> str:
        return self.keys[self.index]

    def call(self, fn):
        """Return fn(key). On 401/402 switch to the next key and retry; raise once every key has failed."""
        while True:
            i = self.index
            try:
                return fn(self.keys[i])
            except Exception as exc:
                code = status_of(exc)
                if code not in SWITCH_ON:
                    raise
                with self._lock:
                    if self.index == i:  # another thread may already have switched
                        if i + 1 >= len(self.keys):
                            raise RuntimeError(f"all {len(self.keys)} API keys are out of credits or invalid") from exc
                        self.index = i + 1
                        print(f"[keys] key {i + 1} got HTTP {code}; switching to key {i + 2}", file=sys.stderr)
