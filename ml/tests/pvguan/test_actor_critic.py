"""Tests for pvguan/actor_critic.py.

Covers:
- Forward shape contracts
- Save/load round-trip (actor head weights preserved bit-identically)
- Checkpoint metadata schema
- Zero-init critic final layer
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import numpy as np
import pytest

from guandan.pvguan.actor_critic import ActorCriticNet, _MLP


# ─── Shape tests ──────────────────────────────────────────────────────────────

def test_score_actions_shape():
    net = ActorCriticNet()
    B = 8
    state = torch.randn(B, 764)
    action = torch.randn(B, 198)
    out = net.score_actions(state, action)
    assert out.shape == (B,), f"Got {out.shape}"


def test_value_shape():
    net = ActorCriticNet()
    B = 8
    sc = torch.randn(B, 875)
    v = net.value(sc)
    assert v.shape == (B,), f"Got {v.shape}"


def test_policy_logits_masks_illegal():
    net = ActorCriticNet()
    K = 5
    state = torch.randn(K, 764)
    action = torch.randn(K, 198)
    mask = torch.ones(K, dtype=torch.bool)
    mask[2] = False

    logits = net.policy_logits(state, action, mask)
    assert logits.shape == (K,)
    assert logits[2].item() < -1e8, "Illegal action not masked"


# ─── Zero-init critic ─────────────────────────────────────────────────────────

def test_critic_zero_init_output():
    """V(s) = 0 at init for any input (final layer zeroed)."""
    net = ActorCriticNet()
    sc = torch.randn(16, 875)
    v = net.value(sc)
    assert torch.allclose(v, torch.zeros(16), atol=1e-6), \
        f"Critic output at init: {v}"


def test_critic_hidden_weights_nonzero():
    """Hidden layers are not all-zero (so gradients flow)."""
    net = ActorCriticNet()
    first_layer_w = net.critic_head.net[0].weight
    assert first_layer_w.abs().max().item() > 0, "Hidden weights should be non-zero"


# ─── Save / load round-trip ───────────────────────────────────────────────────

def test_actor_head_save_load_roundtrip():
    net1 = ActorCriticNet()
    net2 = ActorCriticNet()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        metadata = {
            "actor_dim": 764, "action_dim": 198, "critic_dim": 875,
            "supervisor": "test", "n_samples": 0,
        }
        torch.save({
            "actor_state_dict": net1.actor_head.state_dict(),
            "metadata": metadata,
        }, path)

        from guandan.pvguan.actor_critic import load_warmstart
        returned_meta = load_warmstart(net2, path)

        # Forward pass should be bit-identical
        state  = torch.randn(8, 764)
        action = torch.randn(8, 198)

        with torch.no_grad():
            out1 = net1.score_actions(state, action)
            out2 = net2.score_actions(state, action)

        assert torch.allclose(out1, out2, atol=1e-6), \
            f"Actor outputs differ after load"

        assert returned_meta == metadata


def test_checkpoint_metadata_schema():
    """Checkpoint metadata must contain required keys."""
    required_keys = {
        "actor_dim", "action_dim", "critic_dim", "supervisor",
    }
    net = ActorCriticNet()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        metadata = {
            "actor_dim": 764, "action_dim": 198, "critic_dim": 875,
            "supervisor": "oracle", "n_samples": 500000,
            "val_acc": 0.96, "train_acc": 0.97,
        }
        torch.save({
            "actor_state_dict": net.actor_head.state_dict(),
            "metadata": metadata,
        }, path)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        for k in required_keys:
            assert k in ckpt.get("metadata", {}), f"Missing key: {k}"


# ─── Temperature ─────────────────────────────────────────────────────────────

def test_temperature_scales_logits():
    net1 = ActorCriticNet(temperature=1.0)
    net2 = ActorCriticNet(temperature=2.0)
    # Copy weights so the only difference is temperature
    net2.actor_head.load_state_dict(net1.actor_head.state_dict())

    K = 4
    state  = torch.randn(K, 764)
    action = torch.randn(K, 198)
    mask   = torch.ones(K, dtype=torch.bool)

    logits1 = net1.policy_logits(state, action, mask)
    logits2 = net2.policy_logits(state, action, mask)
    # logits2 should be half of logits1 (divided by τ=2)
    assert torch.allclose(logits1 / 2.0, logits2, atol=1e-5), \
        f"Temperature scaling wrong: {logits1} / 2 != {logits2}"
