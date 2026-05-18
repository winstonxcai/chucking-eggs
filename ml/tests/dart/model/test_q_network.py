from __future__ import annotations

import numpy as np
import pytest
import torch

from guandan.game import GuanDanEnv
from guandan.dart.data.buffer import collate_base_encoded, collate_role_encoded
from guandan.dart.config import QNetConfig
from guandan.dart.model.encoding.base_encoder import StateActionEncoder
from guandan.dart.model.encoding.role_encoder import RoleAwareStateActionEncoder
from guandan.dart.model.q_network import (
    GuanZeroQNet,
    DartQNet,
    DartQNetConfig,
)


def _row_batch(
    encoded: list[dict[str, np.ndarray]],
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    return {
        k: torch.from_numpy(np.stack([row[k] for row in encoded], axis=0))
        .to(device, non_blocking=True)
        for k in encoded[0]
    }


def _build_batch(n: int) -> dict[str, torch.Tensor]:
    env = GuanDanEnv()
    env.reset(seed=0)
    p = env.current_player
    legal = env.legal_moves(p)[:n]
    encoder = StateActionEncoder()
    encoded = encoder.encode_all(env, p, legal)
    return _row_batch(encoded)


def _build_encoded(seed: int, n: int) -> list[dict]:
    env = GuanDanEnv()
    env.reset(seed=seed)
    p = env.current_player
    legal = env.legal_moves(p)[:n]
    encoder = StateActionEncoder()
    return encoder.encode_all(env, p, legal)


def _grouped_from_encoded(groups: list[list[dict]]) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
]:
    flat = [row for group in groups for row in group]
    full = _row_batch(flat)
    state_batch, action_batch, repeats = collate_base_encoded(groups)
    return full, state_batch, action_batch, repeats


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


def test_forward_grouped_matches_repeated_row_forward():
    groups = [_build_encoded(seed=0, n=3), _build_encoded(seed=1, n=5)]
    full, state_batch, action_batch, repeats = _grouped_from_encoded(groups)

    for cfg in (
        QNetConfig(hidden_lstm=32, hidden_mlp=64, n_mlp_layers=2),
        QNetConfig(
            hidden_lstm=32,
            hidden_mlp=64,
            n_mlp_layers=2,
            history_encoder="transformer",
            transformer_nhead=4,
            transformer_layers=1,
            transformer_ff_dim=64,
        ),
    ):
        torch.manual_seed(123)
        net = GuanZeroQNet(cfg).eval()
        with torch.no_grad():
            expected = net(full)
            actual = net.forward_grouped(state_batch, action_batch, repeats)
        assert torch.allclose(actual, expected, atol=1e-6)


def test_forward_grouped_single_state_fast_path_matches_repeated_row_forward():
    groups = [_build_encoded(seed=2, n=6)]
    full, state_batch, action_batch, repeats = _grouped_from_encoded(groups)

    for cfg in (
        QNetConfig(hidden_lstm=32, hidden_mlp=64, n_mlp_layers=2),
        QNetConfig(
            hidden_lstm=32,
            hidden_mlp=64,
            n_mlp_layers=2,
            history_encoder="transformer",
            transformer_nhead=4,
            transformer_layers=1,
            transformer_ff_dim=64,
        ),
    ):
        torch.manual_seed(321)
        net = GuanZeroQNet(cfg).eval()
        with torch.no_grad():
            expected = net(full)
            actual = net.forward_grouped(state_batch, action_batch, repeats)
        assert torch.allclose(actual, expected, atol=1e-6)


# ─── DartQNet ──────────────────────────────────────────────


def _small_dart_cfg() -> DartQNetConfig:
    return DartQNetConfig(
        role_d_model=16,
        history_hidden=16,
        global_hidden=8,
        action_hidden=8,
        trunk_hidden=32,
        trunk_layers=1,
    )


def _build_dart_encoded(seed: int, n: int) -> list[dict]:
    env = GuanDanEnv()
    env.reset(seed=seed)
    p = env.current_player
    legal = env.legal_moves(p)[:n]
    encoder = RoleAwareStateActionEncoder()
    return encoder.encode_all(env, p, legal)


def _dart_row_batch(
    encoded: list[dict[str, np.ndarray]],
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    out = {}
    for k in encoded[0]:
        t = torch.from_numpy(np.stack([row[k] for row in encoded], axis=0)).to(device)
        out[k] = t.long() if k == "trick_head_id" else t.float()
    return out


def test_dart_no_seat_emb():
    """DartQNet must not have a seat_emb attribute or state_dict key."""
    net = DartQNet(_small_dart_cfg())
    assert not hasattr(net, "seat_emb")
    keys = list(net.state_dict().keys())
    assert not any(k.startswith("seat_emb") for k in keys), (
        f"unexpected seat_emb in state_dict: {[k for k in keys if 'seat_emb' in k]}"
    )


def test_dart_forward_shape():
    net = DartQNet(_small_dart_cfg())
    batch = _dart_row_batch(_build_dart_encoded(seed=10, n=5))

    q = net(batch)

    assert q.shape == (5,)
    assert torch.isfinite(q).all()


def test_dart_routing():
    """Heads are gathered by trick_head_id."""
    net = DartQNet(_small_dart_cfg())
    for param in net.parameters():
        torch.nn.init.zeros_(param)
    for head_id, head in enumerate(net.heads):
        head.bias.data.fill_(float(head_id))

    batch = _dart_row_batch(_build_dart_encoded(seed=11, n=4))
    batch["trick_head_id"] = torch.arange(4, dtype=torch.long)

    with torch.no_grad():
        q = net(batch)

    assert torch.equal(q, torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_dart_gradient_flows_to_selected_head_only():
    net = DartQNet(_small_dart_cfg())
    batch = _dart_row_batch(_build_dart_encoded(seed=12, n=4))
    batch["trick_head_id"] = torch.full((4,), 2, dtype=torch.long)

    loss = net(batch).pow(2).mean()
    loss.backward()

    trunk_grads = [p.grad for p in net.trunk.parameters()]
    selected_grads = [p.grad for p in net.heads[2].parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in trunk_grads)
    assert any(g is not None and g.abs().sum() > 0 for g in selected_grads)
    for hid in (0, 1, 3):
        grads = [p.grad for p in net.heads[hid].parameters()]
        assert all(g is None or torch.count_nonzero(g).item() == 0 for g in grads)


def test_dart_forward_grouped_matches_repeated_row_forward():
    groups = [_build_dart_encoded(seed=13, n=3), _build_dart_encoded(seed=14, n=5)]
    flat = [row for group in groups for row in group]
    full = _dart_row_batch(flat)
    state_batch, action_batch, repeats = collate_role_encoded(groups)

    torch.manual_seed(123)
    net = DartQNet(_small_dart_cfg()).eval()
    with torch.no_grad():
        expected = net(full)
        actual = net.forward_grouped(state_batch, action_batch, repeats)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_dart_forward_grouped_single_state():
    groups = [_build_dart_encoded(seed=16, n=7)]
    full = _dart_row_batch(groups[0])
    state_batch, action_batch, repeats = collate_role_encoded(groups)

    torch.manual_seed(456)
    net = DartQNet(_small_dart_cfg()).eval()
    with torch.no_grad():
        expected = net(full)
        actual = net.forward_grouped(state_batch, action_batch, repeats)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_dart_player_blocks_width():
    """DartQNet player_mlp expects 256-wide blocks."""
    net = DartQNet(_small_dart_cfg())
    assert net.player_mlp[0].in_features == 256
