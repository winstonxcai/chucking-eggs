from __future__ import annotations

import tempfile
from pathlib import Path

from guandan.guanzero.config import QNetConfig
from guandan.guanzero.learner import publish_weights
from guandan.guanzero.q_network import init_seat_nets
from guandan.guanzero.worker import maybe_sync_weights


def test_maybe_sync_weights_no_op_when_not_newer():
    q_nets = init_seat_nets(QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2))

    with tempfile.TemporaryDirectory() as td:
        weight_dir = Path(td) / "weights"
        publish_weights(q_nets, weight_dir, version=5)

        result = maybe_sync_weights(q_nets, weight_dir, local_version=5)
        assert result == (5, 0, 0)

        result = maybe_sync_weights(q_nets, weight_dir, local_version=6)
        assert result == (6, 0, 0)


def test_maybe_sync_weights_loads_newer():
    q_nets_pub   = init_seat_nets(QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2))
    q_nets_actor = init_seat_nets(QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2))

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


def test_maybe_sync_weights_lag_gate_defers_load():
    """When max_lag_updates is set, actors keep stale weights until the gap is wide enough."""
    q_nets_pub = init_seat_nets(QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2))
    q_nets_actor = init_seat_nets(QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2))

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
            q_nets_actor, weight_dir, local_version=-1, local_updates=0, max_lag_updates=500,
        )
        assert (v, lu, gu) == (1, 100, 100)

        # Republish at update 300 (lag = 300 - 100 = 200, below threshold 500).
        publish_weights(q_nets_pub, weight_dir, version=2, updates=300)
        v, lu, gu = maybe_sync_weights(
            q_nets_actor, weight_dir, local_version=v, local_updates=lu, max_lag_updates=500,
        )
        # Deferred: version stays at 1, local_updates stays at 100, global_updates is fresh.
        assert (v, lu, gu) == (1, 100, 300)

        # Republish at update 700 (lag = 700 - 100 = 600, exceeds threshold 500).
        publish_weights(q_nets_pub, weight_dir, version=3, updates=700)
        v, lu, gu = maybe_sync_weights(
            q_nets_actor, weight_dir, local_version=v, local_updates=lu, max_lag_updates=500,
        )
        assert (v, lu, gu) == (3, 700, 700)
