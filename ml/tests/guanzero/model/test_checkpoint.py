from __future__ import annotations

import torch

from guandan.guanzero.model.checkpoint import load_checkpoint, save_checkpoint, unwrap_compiled
from guandan.guanzero.config import QNetConfig, TrainConfig
from guandan.guanzero.model.q_network import init_seat_nets


def test_checkpoint_roundtrip_saves_config_and_all_seat_weights(tmp_path):
    cfg = TrainConfig(qnet=QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2))
    q_nets = init_seat_nets(cfg.qnet)
    for param in q_nets[0].parameters():
        param.data.fill_(3.0)

    path = tmp_path / "checkpoints" / "ckpt.pt"
    save_checkpoint(path, q_nets, cfg, episode_or_update=12)

    ckpt = load_checkpoint(path)
    assert ckpt["episode"] == 12
    assert ckpt["config"]["qnet"]["hidden_lstm"] == 16
    assert set(ckpt["q_nets"]) == {0, 1, 2, 3}

    loaded = init_seat_nets(cfg.qnet)
    loaded[0].load_state_dict(ckpt["q_nets"][0])
    for param in loaded[0].parameters():
        assert torch.all(param == 3.0)


def test_unwrap_compiled_returns_original_module_when_present():
    net = torch.nn.Linear(2, 1)

    class Wrapper:
        _orig_mod = net

    assert unwrap_compiled(Wrapper()) is net
    assert unwrap_compiled(net) is net
