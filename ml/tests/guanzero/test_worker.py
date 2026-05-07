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
        assert result == 5

        result = maybe_sync_weights(q_nets, weight_dir, local_version=6)
        assert result == 6


def test_maybe_sync_weights_loads_newer():
    q_nets_pub   = init_seat_nets(QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2))
    q_nets_actor = init_seat_nets(QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2))

    for net in q_nets_pub.values():
        for p in net.parameters():
            p.data.fill_(99.0)

    with tempfile.TemporaryDirectory() as td:
        weight_dir = Path(td) / "weights"
        publish_weights(q_nets_pub, weight_dir, version=3)

        new_ver = maybe_sync_weights(q_nets_actor, weight_dir, local_version=-1)
        assert new_ver == 3

        for p in range(4):
            for param in q_nets_actor[p].parameters():
                assert (param.data == 99.0).all()
