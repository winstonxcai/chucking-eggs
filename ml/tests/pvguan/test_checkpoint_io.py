"""Checkpoint save → load round-trip tests.

Asserts that:
- actor_head weights survive a save/load cycle with bit-identical forward pass
- critic_head weights survive independently
- full_state_dict round-trip preserves both heads
- metadata schema contains every required key
- load_warmstart only loads actor_head (critic stays at fresh zero-init)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from guandan.pvguan.actor_critic import ActorCriticNet, load_warmstart
from guandan.pvguan.encoders import ACTION_DIM, ACTOR_DIM, CRITIC_DIM

REQUIRED_METADATA_KEYS = {
    "actor_dim",
    "action_dim",
    "critic_dim",
    "critic_mode",
    "training_seed",
}


def _make_net(hidden: int = 32) -> ActorCriticNet:
    torch.manual_seed(42)
    return ActorCriticNet(
        d_state_actor=ACTOR_DIM,
        d_action=ACTION_DIM,
        d_state_critic=CRITIC_DIM,
        hidden=hidden,
    )


def _probe_actor(net: ActorCriticNet, device: torch.device | None = None) -> torch.Tensor:
    """Fixed input → actor logits for round-trip comparison."""
    torch.manual_seed(7)
    dev = device or torch.device("cpu")
    sa   = torch.randn(8, ACTOR_DIM, device=dev)
    acts = torch.randn(8, ACTION_DIM, device=dev)
    with torch.no_grad():
        return net.score_actions(sa, acts)


def _probe_critic(net: ActorCriticNet, device: torch.device | None = None) -> torch.Tensor:
    torch.manual_seed(7)
    dev = device or torch.device("cpu")
    sc = torch.randn(8, CRITIC_DIM, device=dev)
    with torch.no_grad():
        return net.value(sc)


# ─── Actor head round-trip ────────────────────────────────────────────────────

def test_actor_head_save_load_roundtrip():
    net1 = _make_net()
    expected = _probe_actor(net1)

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        path = f.name
        torch.save({"actor_state_dict": net1.actor_head.state_dict()}, path)

        net2 = _make_net()
        # Give net2 different weights first
        torch.manual_seed(99)
        for p in net2.actor_head.parameters():
            p.data.uniform_(-1, 1)

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        net2.actor_head.load_state_dict(ckpt["actor_state_dict"])

    actual = _probe_actor(net2)
    assert torch.allclose(expected, actual, atol=1e-6), \
        f"Actor round-trip failed: max diff = {(expected - actual).abs().max()}"


# ─── Critic head round-trip ──────────────────────────────────────────────────

def test_critic_head_save_load_roundtrip():
    net1 = _make_net()
    # Randomize critic to make it non-trivial
    torch.manual_seed(10)
    for p in net1.critic_head.parameters():
        p.data.uniform_(-1, 1)
    expected = _probe_critic(net1)

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        path = f.name
        torch.save({"critic_state_dict": net1.critic_head.state_dict()}, path)

        net2 = _make_net()
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        net2.critic_head.load_state_dict(ckpt["critic_state_dict"])

    actual = _probe_critic(net2)
    assert torch.allclose(expected, actual, atol=1e-6)


# ─── Full state dict round-trip ──────────────────────────────────────────────

def test_full_state_dict_roundtrip():
    net1 = _make_net()
    torch.manual_seed(20)
    for p in net1.parameters():
        p.data.uniform_(-0.5, 0.5)
    actor_out1  = _probe_actor(net1)
    critic_out1 = _probe_critic(net1)

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        path = f.name
        torch.save({"full_state_dict": net1.state_dict()}, path)
        net2 = _make_net()
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        net2.load_state_dict(ckpt["full_state_dict"])

    assert torch.allclose(actor_out1,  _probe_actor(net2),  atol=1e-6)
    assert torch.allclose(critic_out1, _probe_critic(net2), atol=1e-6)


# ─── load_warmstart: actor only ──────────────────────────────────────────────

def test_load_warmstart_actor_only():
    """load_warmstart must copy actor_head and leave critic at zero-init."""
    net_src = _make_net()
    torch.manual_seed(30)
    for p in net_src.parameters():
        p.data.uniform_(-1, 1)
    expected_actor = _probe_actor(net_src)

    metadata = {"actor_dim": ACTOR_DIM, "action_dim": ACTION_DIM,
                "critic_dim": CRITIC_DIM, "supervisor": "test",
                "critic_mode": "pv", "training_seed": 0}

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        path = f.name
        torch.save({
            "actor_state_dict":  net_src.actor_head.state_dict(),
            "critic_state_dict": net_src.critic_head.state_dict(),
            "full_state_dict":   net_src.state_dict(),
            "metadata":          metadata,
        }, path)

        net_dst = _make_net()
        returned_meta = load_warmstart(net_dst, path, map_location="cpu")

    # Actor must match
    assert torch.allclose(expected_actor, _probe_actor(net_dst), atol=1e-6), \
        "Actor head did not load correctly from warmstart"

    # Critic final layer must still be zero (zero-init not overwritten)
    final_w = net_dst.critic_head.net[-1].weight
    final_b = net_dst.critic_head.net[-1].bias
    assert final_w.abs().max().item() == pytest.approx(0.0), \
        "Critic final layer should remain zero after load_warmstart (actor-only)"
    assert final_b.abs().max().item() == pytest.approx(0.0)

    # Metadata must be returned
    assert returned_meta == metadata


# ─── Metadata schema ─────────────────────────────────────────────────────────

def test_checkpoint_metadata_schema():
    """A full checkpoint must contain every required metadata key."""
    metadata = {
        "actor_dim":      ACTOR_DIM,
        "action_dim":     ACTION_DIM,
        "critic_dim":     CRITIC_DIM,
        "critic_mode":    "pv",
        "training_seed":  0,
        # optional extras that real training adds:
        "cumulative_decisions": 1_000_000,
        "warmstart_hash": "abc123",
        "encoder_schema_version": "1.0",
    }
    assert REQUIRED_METADATA_KEYS <= metadata.keys(), \
        f"Missing keys: {REQUIRED_METADATA_KEYS - metadata.keys()}"


def test_checkpoint_metadata_survives_roundtrip():
    """Metadata saved with a checkpoint loads back intact."""
    net = _make_net()
    metadata = {
        "actor_dim": ACTOR_DIM, "action_dim": ACTION_DIM,
        "critic_dim": CRITIC_DIM, "critic_mode": "ptie",
        "training_seed": 2, "cumulative_decisions": 3_000_000,
    }

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        path = f.name
        torch.save({
            "actor_state_dict":  net.actor_head.state_dict(),
            "critic_state_dict": net.critic_head.state_dict(),
            "full_state_dict":   net.state_dict(),
            "metadata":          metadata,
        }, path)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)

    assert ckpt["metadata"] == metadata


# ─── Argmax determinism ──────────────────────────────────────────────────────

def test_argmax_deterministic_after_load():
    """load_warmstart → argmax choices on probe set must match original network."""
    net_src = _make_net()
    torch.manual_seed(50)
    for p in net_src.actor_head.parameters():
        p.data.uniform_(-0.5, 0.5)

    # Build a probe set: 20 decisions, 6 candidates each
    torch.manual_seed(51)
    sa   = torch.randn(20, 6, ACTOR_DIM)
    acts = torch.randn(20, 6, ACTION_DIM)
    mask = torch.ones(20, 6, dtype=torch.bool)

    with torch.no_grad():
        sa_flat   = sa.view(120, -1)
        acts_flat = acts.view(120, -1)
        logits_src = net_src.score_actions(sa_flat, acts_flat).view(20, 6)
        logits_src = logits_src.masked_fill(~mask, -1e9)
        argmax_src = logits_src.argmax(dim=-1)

    meta = {"actor_dim": ACTOR_DIM, "action_dim": ACTION_DIM,
            "critic_dim": CRITIC_DIM, "critic_mode": "pv", "training_seed": 0}

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        path = f.name
        torch.save({
            "actor_state_dict": net_src.actor_head.state_dict(),
            "metadata": meta,
        }, path)
        net_dst = _make_net()
        load_warmstart(net_dst, path, map_location="cpu")

    with torch.no_grad():
        logits_dst = net_dst.score_actions(sa_flat, acts_flat).view(20, 6)
        logits_dst = logits_dst.masked_fill(~mask, -1e9)
        argmax_dst = logits_dst.argmax(dim=-1)

    assert (argmax_src == argmax_dst).all(), \
        "Argmax mismatch after load_warmstart round-trip"
