from __future__ import annotations

import tempfile
from pathlib import Path

from guandan.guanzero.config import QNetConfig
from guandan.guanzero.learner import load_latest_weights, publish_weights
from guandan.guanzero.q_network import init_seat_nets


def test_publish_and_load_weights():
    q_nets = init_seat_nets(QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2))

    with tempfile.TemporaryDirectory() as td:
        weight_dir = Path(td) / "weights"

        ver, sds = load_latest_weights(weight_dir)
        assert ver is None and sds is None

        publish_weights(q_nets, weight_dir, version=0)

        assert (weight_dir / "latest.txt").exists()
        assert (weight_dir / "weights_0.pt").exists()
        assert not (weight_dir / "weights_0.tmp").exists()

        ver, sds = load_latest_weights(weight_dir)
        assert ver == 0
        assert set(sds.keys()) == {0, 1, 2, 3}

        publish_weights(q_nets, weight_dir, version=1)
        ver2, _ = load_latest_weights(weight_dir)
        assert ver2 == 1
        assert (weight_dir / "weights_1.pt").exists()
