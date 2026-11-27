"""#58: the runner with several workers over the real gateway client and a fake gateway. No sockets, no keys."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("galaxus.model.apertus.api", reason="needs the gateway client (#13, #38)")

MODULE_PATH = Path(__file__).resolve().parents[2] / "bench" / "gateway_load.py"
spec = importlib.util.spec_from_file_location("galaxus_bench_gateway_load", MODULE_PATH)
load = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = load
spec.loader.exec_module(load)

KEYS = ("secret-key-ONE-1234", "secret-key-TWO-5678")


def lines(root, exp="LOAD"):
    path = root / "predictions" / exp / "dev.jsonl"
    return [json.loads(line)["image_id"] for line in path.read_text(encoding="utf-8").splitlines()]


def test_many_workers_give_every_line_once(tmp_path, galaxus_data):
    gateway = load.FakeGateway(latency=0.002)
    report = load.run_load(gateway, 120, workers=8, batch_size=3, root=tmp_path, keys=KEYS)
    ids = lines(tmp_path)
    assert report.complete and len(ids) == 120 == len(set(ids))


def test_429_bursts_and_5xx_are_absorbed_with_four_retries(tmp_path, galaxus_data):
    gateway = load.FakeGateway(burst_429_every=10, burst_429_length=2, error_5xx_every=13)
    report = load.run_load(gateway, 100, workers=6, batch_size=2, root=tmp_path, keys=KEYS, retries=4)
    assert report.complete and len(set(lines(tmp_path))) == 100
    assert gateway.statuses[429] > 0 and gateway.statuses[503] > 0


def test_finding_a_burst_longer_than_the_retries_stops_the_run_and_the_restart_finishes(tmp_path, galaxus_data):
    """Deterministic: one worker, so all three 429s of a burst hit the same request; 2 retries run out."""
    gateway = load.FakeGateway(burst_429_every=10, burst_429_length=3)
    with pytest.raises(Exception, match="HTTP 429"):
        load.run_load(gateway, 100, workers=1, batch_size=2, root=tmp_path, keys=KEYS, retries=2)
    report = load.run_load(load.FakeGateway(), 100, workers=6, batch_size=2, root=tmp_path, keys=KEYS, retries=4)
    assert report.complete and len(lines(tmp_path)) == 100 == len(set(lines(tmp_path)))


def test_a_402_switches_the_key_once_for_every_worker_and_no_key_is_printed(tmp_path, galaxus_data, capsys):
    gateway = load.FakeGateway(latency=0.001, key_budget={KEYS[0]: 30})
    report = load.run_load(gateway, 90, workers=8, batch_size=2, root=tmp_path, keys=KEYS)
    out = capsys.readouterr()
    assert report.complete and len(set(lines(tmp_path))) == 90
    assert gateway.keys_used[KEYS[0]] == 30 and gateway.keys_used[KEYS[1]] == 60
    assert (out.out + out.err).count("switching to key 2") == 1
    written = "".join(p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.jsonl"))
    for key in KEYS:
        assert key not in out.out + out.err + written


def test_a_dead_gateway_stops_the_run_and_a_restart_completes_without_duplicates(tmp_path, galaxus_data):
    with pytest.raises(Exception, match="HTTP 503"):
        load.run_load(load.FakeGateway(dead_after=40), 100, workers=4, batch_size=2, root=tmp_path, keys=KEYS)
    first = lines(tmp_path)
    assert 0 < len(first) < 100
    report = load.run_load(load.FakeGateway(), 100, workers=4, batch_size=2, root=tmp_path, keys=KEYS)
    ids = lines(tmp_path)
    assert report.complete and report.already_done == len(first) and len(ids) == 100 == len(set(ids))


def test_more_workers_are_faster_over_a_slow_gateway(tmp_path, galaxus_data):
    slow = load.run_load(load.FakeGateway(latency=0.02), 40, workers=1, batch_size=2, root=tmp_path / "a", keys=KEYS)
    fast = load.run_load(load.FakeGateway(latency=0.02), 40, workers=8, batch_size=2, root=tmp_path / "b", keys=KEYS)
    assert fast.images_per_second > 3 * slow.images_per_second
