"""#58: load test of ``runner.run(workers=N)`` with the real gateway client over a **fake** gateway.

Nothing here touches the network or a real key. A :class:`FakeGateway` stands in for the Hack Apertus gateway as the
``transport=`` of :class:`galaxus.model.apertus.api.GatewayBackend` (the real client, with its real retries and the
real :class:`galaxus.keys.KeyRing`), and can add latency, HTTP 429 bursts, 5xx errors and a key that runs out of
credits (402) mid-run. The serve runner drives it with N workers, exactly as on Saturday.

What it proves (tests in ``tests/serve/test_gateway_load.py``): no duplicate or missing lines with several workers;
a run killed by a dead gateway resumes and completes; a 402 switches the key once for all workers, with no key in
any output; 429s and 5xx within the client's retries are absorbed, not failures.

Finding: a 429 burst longer than the client's retries (default 2), or a burst followed by a 5xx on the same request,
exhausts them, and the whole run stops (it resumes on restart). With ``retries=4`` the tested load with 6 workers is
absorbed, so the Saturday predictor should build its ``GatewayBackend`` with ``retries=4`` (see the run sheet).

    python bench/gateway_load.py --images 200 --latency 0.5 --workers 1,4,8,16
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image as PillowImage


def valid_answer() -> dict[str, str]:
    """One allowed class per dimension, from ``galaxus.data.classes`` (#82: no dimension or class typed here)."""
    from galaxus.data.classes import allowed_classes, dimensions

    allowed = allowed_classes()
    return {d: min(allowed[d]) for d in dimensions()}


def tiny_jpeg() -> bytes:
    """A 4 x 4 RGB JPEG: a real image for the client to encode, never a competition image."""
    buffer = io.BytesIO()
    PillowImage.new("RGB", (4, 4), (200, 30, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


@dataclass
class FakeGateway:
    """An OpenAI-compatible endpoint in memory. Thread-safe; records what it saw, never prints a key."""

    latency: float = 0.0
    burst_429_every: int = 0      # every n-th request starts a burst ...
    burst_429_length: int = 0     # ... of this many 429 answers
    error_5xx_every: int = 0      # every n-th request answers 503 once
    key_budget: dict[str, int] = field(default_factory=dict)  # key -> successful calls before it answers 402
    dead_after: int | None = None  # every request after this many answers 503 (a gateway that went down)
    requests: int = 0
    statuses: Counter = field(default_factory=Counter)
    keys_used: Counter = field(default_factory=Counter)
    in_flight: int = 0            # calls being answered right now
    max_in_flight: int = 0        # the most calls answered at the same time (#140: overlap, not wall time)
    _burst_left: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _answer: str = ""

    def __post_init__(self) -> None:
        # Read the classes once: resolving them per request reads files and would hide the workers' speed-up.
        self._answer = json.dumps(valid_answer())

    def __call__(self, url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, bytes]:
        key = headers.get("Authorization", "").removeprefix("Bearer ")
        with self._lock:
            self.requests += 1
            number = self.requests
            if self.dead_after is not None and number > self.dead_after:
                status = 503
            elif self._burst_left > 0:
                self._burst_left -= 1
                status = 429
            elif self.burst_429_every and number % self.burst_429_every == 0:
                self._burst_left = max(0, self.burst_429_length - 1)
                status = 429
            elif self.error_5xx_every and number % self.error_5xx_every == 0:
                status = 503
            elif key in self.key_budget and self.key_budget[key] <= 0:
                status = 402
            else:
                status = 200
                if key in self.key_budget:
                    self.key_budget[key] -= 1
                self.keys_used[key] += 1
            self.statuses[status] += 1
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.latency:
                time.sleep(self.latency)
        finally:
            with self._lock:
                self.in_flight -= 1
        if status != 200:
            return status, json.dumps({"error": {"code": status}}).encode()
        reply = {"choices": [{"message": {"content": self._answer}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 1100, "completion_tokens": 40}, "model": "fake-apertus"}
        return 200, json.dumps(reply).encode()


class GatewayPredictor:
    """The contract over ``GatewayBackend``: one request per image, the JSON answer parsed into the six labels.

    Stand-in for the zero-shot predictor (#47) so the load test does not wait on it; same client, same calls.
    """

    def __init__(self, backend: Any, image: bytes | None = None) -> None:
        self.backend = backend
        self.image = image or tiny_jpeg()

    def predict(self, items: Sequence[Any]) -> list[dict[str, Any]]:
        answers = []
        for item in items:
            generation = self.backend.generate([{"role": "user", "content": "Classify this product image."}],
                                               [self.image])
            labels = json.loads(generation.text)
            answers.append({"image_id": item.image_id, "labels": labels, "confidence": None})
        return answers


@dataclass(frozen=True)
class Item:
    image_id: str
    path: str = ""
    context: None = None


def make_backend(gateway: FakeGateway, keys: Sequence[str], backoff: float = 0.0, retries: int = 2) -> Any:
    from galaxus.keys import KeyRing
    from galaxus.model.apertus.api import GatewayBackend

    return GatewayBackend("http://fake-gateway/v1", "fake-apertus", keys=KeyRing(list(keys)), transport=gateway,
                          retries=retries, backoff=backoff)


def run_load(
    gateway: FakeGateway, images: int, workers: int, batch_size: int, root: Path, exp: str = "LOAD",
    keys: Sequence[str] = ("fake-key-one", "fake-key-two"), backoff: float = 0.0, retries: int = 2,
) -> Any:
    from galaxus.serve.runner import run

    predictor = GatewayPredictor(make_backend(gateway, keys, backoff, retries))
    items = [Item(f"img_{i:05d}") for i in range(images)]
    return run(predictor, items, exp, "dev", batch_size=batch_size, root=root, log=None, workers=workers)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python bench/gateway_load.py", description=__doc__.split("\n")[0])
    parser.add_argument("--images", type=int, default=200)
    parser.add_argument("--latency", type=float, default=0.5, help="seconds per request (a guess until #46 measures)")
    parser.add_argument("--workers", default="1,4,8,16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--burst-every", type=int, default=50, help="a 429 burst every n requests (0: none)")
    parser.add_argument("--retries", type=int, default=4, help="GatewayBackend retries per request")
    arguments = parser.parse_args(argv)

    print("| workers | img/s | 429s absorbed | 5xx absorbed | wall s |\n| --- | --- | --- | --- | --- |")
    for workers in [int(w) for w in arguments.workers.split(",")]:
        gateway = FakeGateway(latency=arguments.latency, burst_429_every=arguments.burst_every, burst_429_length=2,
                              error_5xx_every=97)
        with tempfile.TemporaryDirectory() as folder:
            started = time.perf_counter()
            report = run_load(gateway, arguments.images, workers, arguments.batch_size, Path(folder),
                              retries=arguments.retries)
            wall = time.perf_counter() - started
        print(f"| {workers} | {report.images_per_second} | {gateway.statuses[429]} | {gateway.statuses[503]} | "
              f"{wall:.1f} |")
    print("\nSimulated latency only: the real gateway's latency and rate limit come from #46 / #59.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
