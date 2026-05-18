from __future__ import annotations

import errno
import tempfile
from pathlib import Path

from guandan.dart.config import QNetConfig
from guandan.dart.data.returns import EpisodeTags, TrainSample
from guandan.dart.runtime.actor import LaneConfig, all_latest_seats
from guandan.dart.runtime.learner import publish_weights
from guandan.dart.model.q_network import init_guanzero_nets
from guandan.dart.runtime.actor.runtime import maybe_sync_weights
from guandan.dart.runtime.actor.samples import (
    ActorSampleAccumulator,
    QueueBatchMeta,
)
from guandan.dart.runtime.worker import (
    _is_closed_queue_error,
)


def _sample(player: int, value: float) -> TrainSample:
    return TrainSample(
        player=player,
        encoded={
            "state": value,
            "action": value + 1,
        },
        mc_return=value,
        phase_self=1,
        trick_role=2,
        phase_partner=3,
        action_type=4,
        is_pass=0,
        is_bomb=0,
        bomb_available=1,
        num_legal_actions=5,
        q_gap=0.25,
        chosen_by_epsilon=0,
        episode_mode=0,
        opponent_id=0,
        latest_team=0,
        terminal_reward=3.0,
    )


def test_actor_sample_accumulator_stacks_and_pops_batches():
    acc = ActorSampleAccumulator(channel_keys=("state", "action"), include_players=True)
    lane = LaneConfig(seed=1, seats=all_latest_seats(0.0), tags=EpisodeTags())

    acc.append_lane(lane, [_sample(0, 1.0), _sample(2, 2.0), _sample(1, 3.0)])
    msg = acc.pop_message(
        2,
        QueueBatchMeta(actor_id=7, version=3, local_updates=100, global_updates=120),
    )

    assert msg is not None
    assert msg["actor_id"] == 7
    assert msg["version"] == 3
    assert msg["local_updates"] == 100
    assert msg["global_updates"] == 120
    assert msg["stacked"]["state"].tolist() == [1.0, 2.0]
    assert msg["stacked"]["action"].tolist() == [2.0, 3.0]
    assert msg["players"].tolist() == [0, 2]
    assert msg["returns"].tolist() == [1.0, 2.0]
    assert msg["num_legal_actions"].tolist() == [5, 5]
    assert len(acc) == 1


def test_queue_error_classifier_only_accepts_closed_queue_errors():
    assert _is_closed_queue_error(BrokenPipeError())
    assert _is_closed_queue_error(OSError(errno.EBADF, "bad file descriptor"))
    assert _is_closed_queue_error(ValueError("Queue is closed"))
    assert not _is_closed_queue_error(OSError(errno.EIO, "I/O error"))
    assert not _is_closed_queue_error(ValueError("bad payload"))


def test_maybe_sync_weights_no_op_when_not_newer():
    q_nets = init_guanzero_nets(QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2))

    with tempfile.TemporaryDirectory() as td:
        weight_dir = Path(td) / "weights"
        publish_weights(q_nets, weight_dir, version=5)

        result = maybe_sync_weights(q_nets, weight_dir, local_version=5)
        assert result == (5, 0, 0)

        result = maybe_sync_weights(q_nets, weight_dir, local_version=6)
        assert result == (6, 0, 0)


def test_maybe_sync_weights_loads_newer():
    q_nets_pub   = init_guanzero_nets(QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2))
    q_nets_actor = init_guanzero_nets(QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2))

    for net in q_nets_pub.values():
        for p in net.parameters():
            p.data.fill_(99.0)

    with tempfile.TemporaryDirectory() as td:
        weight_dir = Path(td) / "weights"
        publish_weights(q_nets_pub, weight_dir, version=3)

        new_ver, local_updates, global_updates = maybe_sync_weights(
            q_nets_actor, weight_dir, local_version=-1,
        )
        assert new_ver == 3
        assert local_updates == 0
        assert global_updates == 0

        for p in range(4):
            for param in q_nets_actor[p].parameters():
                assert (param.data == 99.0).all()


def test_maybe_sync_weights_interval_gate_defers_load():
    """When sync_interval_updates is set, actors keep stale weights until the gap is wide enough."""
    q_nets_pub = init_guanzero_nets(QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2))
    q_nets_actor = init_guanzero_nets(QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2))

    # Make published weights distinguishable from actor's initial weights.
    for net in q_nets_pub.values():
        for p in net.parameters():
            p.data.fill_(7.0)

    with tempfile.TemporaryDirectory() as td:
        weight_dir = Path(td) / "weights"

        # Initial publish at update 100. Actor has never synced (-1).
        publish_weights(q_nets_pub, weight_dir, version=1, updates=100)

        # First sync ignores the lag gate (local_version=-1) — must load.
        v, lu, gu = maybe_sync_weights(
            q_nets_actor, weight_dir, local_version=-1, local_updates=0, sync_interval_updates=500,
        )
        assert (v, lu, gu) == (1, 100, 100)

        # Republish at update 300 (lag = 300 - 100 = 200, below threshold 500).
        publish_weights(q_nets_pub, weight_dir, version=2, updates=300)
        v, lu, gu = maybe_sync_weights(
            q_nets_actor, weight_dir, local_version=v, local_updates=lu, sync_interval_updates=500,
        )
        # Deferred: version stays at 1, local_updates stays at 100, global_updates is fresh.
        assert (v, lu, gu) == (1, 100, 300)

        # Republish at update 700 (lag = 700 - 100 = 600, exceeds threshold 500).
        publish_weights(q_nets_pub, weight_dir, version=3, updates=700)
        v, lu, gu = maybe_sync_weights(
            q_nets_actor, weight_dir, local_version=v, local_updates=lu, sync_interval_updates=500,
        )
        assert (v, lu, gu) == (3, 700, 700)
