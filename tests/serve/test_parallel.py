"""#36: several batches in flight. Synthetic items, no network: a stub that sleeps stands in for an HTTP backend."""

from __future__ import annotations

import json
import threading
import time

import pytest
from serve_helpers import DIMENSIONS, Item, Pred

from galaxus.serve import bench, runner


class SlowPredictor:
    """Waits like a network call, records how many batches overlap, and can fail on chosen ids."""

    def __init__(self, delay: float = 0.05, fail_on: frozenset[str] = frozenset()) -> None:
        self.delay = delay
        self.fail_on = fail_on
        self.active = 0
        self.max_active = 0
        self.seen: list[str] = []
        self.lock = threading.Lock()

    def predict(self, items):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            if self.fail_on & {item.image_id for item in items}:
                raise ConnectionError("simulated HTTP failure")
            with self.lock:
                self.seen.extend(item.image_id for item in items)
            return [Pred(item.image_id, {d: "x" for d in DIMENSIONS}) for item in items]
        finally:
            with self.lock:
                self.active -= 1


def make_items(count):
    return [Item(f"img_{index:03d}", f"/nowhere/{index}.jpg") for index in range(count)]


def ids_in(path):
    return [json.loads(line)["image_id"] for line in path.read_text(encoding="utf-8").splitlines()]


def test_workers_keep_several_batches_in_flight(tmp_path):
    items = make_items(32)
    predictor = SlowPredictor()
    report = runner.run(predictor, items, "E", "dev", batch_size=4, root=tmp_path, log=None, workers=4)
    path = runner.predictions_path("E", "dev", tmp_path)
    assert report.complete and report.workers == 4
    assert predictor.max_active >= 2
    assert sorted(ids_in(path)) == [item.image_id for item in items]


def test_default_stays_sequential(tmp_path):
    predictor = SlowPredictor(delay=0.01)
    runner.run(predictor, make_items(12), "E", "dev", batch_size=3, root=tmp_path, log=None)
    assert predictor.max_active == 1


def test_a_failing_batch_keeps_the_finished_ones_and_a_restart_completes(tmp_path):
    items = make_items(24)
    with pytest.raises(ConnectionError):
        runner.run(SlowPredictor(fail_on=frozenset({"img_013"})), items, "E", "dev", batch_size=2,
                   root=tmp_path, log=None, workers=3)
    path = runner.predictions_path("E", "dev", tmp_path)
    first = ids_in(path)
    assert first and "img_013" not in first and len(first) == len(set(first))

    second = SlowPredictor(delay=0.0)
    report = runner.run(second, items, "E", "dev", batch_size=2, root=tmp_path, log=None, workers=3)
    final = ids_in(path)
    assert report.complete and sorted(final) == [item.image_id for item in items] and len(final) == len(set(final))
    assert set(second.seen).isdisjoint(first)


def test_parallel_is_faster_over_a_slow_backend(tmp_path):
    items = make_items(16)
    started = time.perf_counter()
    runner.run(SlowPredictor(delay=0.1), items, "S", "dev", batch_size=2, root=tmp_path, log=None)
    sequential = time.perf_counter() - started
    started = time.perf_counter()
    runner.run(SlowPredictor(delay=0.1), items, "P", "dev", batch_size=2, root=tmp_path, log=None, workers=8)
    parallel = time.perf_counter() - started
    assert parallel < sequential / 2


def test_bench_with_workers_times_every_batch():
    predictor = SlowPredictor(delay=0.01)
    result = bench.bench(predictor, make_items(20), batch_size=2, warmup_batches=1, workers=4)
    assert result.images == 18 and len(predictor.seen) == 20 and predictor.max_active >= 2


def test_bad_worker_counts_are_refused(tmp_path):
    with pytest.raises(ValueError, match="workers"):
        runner.run(SlowPredictor(), make_items(2), "E", "dev", root=tmp_path, log=None, workers=0)
    with pytest.raises(ValueError, match="workers"):
        bench.bench(SlowPredictor(), make_items(4), batch_size=1, workers=0)
