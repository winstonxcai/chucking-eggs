from __future__ import annotations

import queue

from guandan.dart.runtime.learner import _drain_sample_queue


class _FakeAdapter:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def drain_message(self, msg: dict) -> int:
        self.messages.append(msg)
        return int(msg["samples"])


def test_drain_sample_queue_caps_batches_and_counts_samples():
    sample_queue: queue.Queue[dict] = queue.Queue()
    for samples in (3, 5, 7):
        sample_queue.put({"samples": samples})
    adapter = _FakeAdapter()

    drained_batches, drained_samples = _drain_sample_queue(
        sample_queue, adapter, max_batches=2
    )

    assert drained_batches == 2
    assert drained_samples == 8
    assert [msg["samples"] for msg in adapter.messages] == [3, 5]
    assert sample_queue.qsize() == 1


def test_drain_sample_queue_captures_latest_actor_rng_state():
    sample_queue: queue.Queue[dict] = queue.Queue()
    sample_queue.put({"actor_id": 2, "samples": 3, "actor_rng_state": {"n": 1}})
    sample_queue.put({"actor_id": 2, "samples": 4, "actor_rng_state": {"n": 2}})
    adapter = _FakeAdapter()
    actor_rng_states: dict[int, dict] = {}

    drained_batches, drained_samples = _drain_sample_queue(
        sample_queue, adapter, max_batches=4, actor_rng_states=actor_rng_states
    )

    assert drained_batches == 2
    assert drained_samples == 7
    assert actor_rng_states == {2: {"n": 2}}


def test_drain_sample_queue_captures_actor_lag_metadata():
    sample_queue: queue.Queue[dict] = queue.Queue()
    sample_queue.put({
        "actor_id": 1,
        "version": 3,
        "local_updates": 120,
        "global_updates": 150,
        "samples": 5,
    })
    adapter = _FakeAdapter()
    actor_stats: dict[int, dict[str, int]] = {}

    drained_batches, drained_samples = _drain_sample_queue(
        sample_queue, adapter, max_batches=4, actor_stats=actor_stats
    )

    assert drained_batches == 1
    assert drained_samples == 5
    assert actor_stats == {
        1: {"version": 3, "local_updates": 120, "global_updates": 150}
    }


def test_drain_sample_queue_propagates_unexpected_errors():
    sample_queue: queue.Queue[dict] = queue.Queue()
    sample_queue.put({"samples": "bad"})
    adapter = _FakeAdapter()

    try:
        _drain_sample_queue(sample_queue, adapter, max_batches=1)
    except ValueError:
        pass
    else:
        raise AssertionError("unexpected drain errors should propagate")


def test_drain_sample_queue_empty_returns_zero():
    sample_queue: queue.Queue[dict] = queue.Queue()
    adapter = _FakeAdapter()

    drained_batches, drained_samples = _drain_sample_queue(
        sample_queue, adapter, max_batches=4
    )

    assert drained_batches == 0
    assert drained_samples == 0
    assert adapter.messages == []
