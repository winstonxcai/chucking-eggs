from __future__ import annotations

import torch

from guandan.game import GuanDanEnv
from guandan.guanzero.buffer import collate_encoded
from guandan.guanzero.config import QNetConfig
from guandan.guanzero.encoder import StateActionEncoder
from guandan.guanzero.q_network import GuanZeroQNet


def _build_batch(n: int) -> dict[str, torch.Tensor]:
    env = GuanDanEnv()
    env.reset(seed=0)
    p = env.current_player
    legal = env.legal_moves(p)[:n]
    encoder = StateActionEncoder()
    encoded = encoder.encode_all(env, p, legal)
    return collate_encoded(encoded)


def test_forward_returns_one_q_per_candidate():
    net = GuanZeroQNet(QNetConfig(hidden_lstm=32, hidden_mlp=64, n_mlp_layers=2))
    batch = _build_batch(n=5)
    q = net(batch)
    assert q.shape == (5,)
    assert torch.isfinite(q).all()


def test_gradient_flows_through_history_module_and_mlp():
    net = GuanZeroQNet(QNetConfig(hidden_lstm=32, hidden_mlp=64, n_mlp_layers=2))
    batch = _build_batch(n=4)
    target = torch.zeros(4)
    q = net(batch)
    loss = ((q - target) ** 2).mean()
    loss.backward()

    hist_grads = [p.grad for p in net.history_module.parameters()]
    mlp_grads = [p.grad for p in net.mlp.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in hist_grads)
    assert all(g is not None and torch.isfinite(g).all() for g in mlp_grads)
    assert any(g.abs().sum() > 0 for g in hist_grads)
    assert any(g.abs().sum() > 0 for g in mlp_grads)


def test_gradient_flows_through_transformer_and_mlp():
    net = GuanZeroQNet(QNetConfig(
        hidden_lstm=32, hidden_mlp=64, n_mlp_layers=2,
        history_encoder="transformer", transformer_nhead=4,
        transformer_layers=1, transformer_ff_dim=64,
    ))
    batch = _build_batch(n=4)
    target = torch.zeros(4)
    q = net(batch)
    loss = ((q - target) ** 2).mean()
    loss.backward()

    hist_grads = [p.grad for p in net.history_module.parameters()]
    mlp_grads = [p.grad for p in net.mlp.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in hist_grads)
    assert all(g is not None and torch.isfinite(g).all() for g in mlp_grads)
    assert any(g.abs().sum() > 0 for g in hist_grads)
    assert any(g.abs().sum() > 0 for g in mlp_grads)
