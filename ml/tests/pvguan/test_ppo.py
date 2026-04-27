"""PPO tests.

Covers: ratio/clip math, masked-softmax legality, entropy > 0,
KL direction (π_warm || π_θ) and schedule (frozen during warmup → decay),
gradient flow (actor frozen during critic warmup).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from guandan.pvguan.actor_critic import ActorCriticNet
from guandan.pvguan.buffer import Decision, PlayerTrack, RolloutBuffer
from guandan.pvguan.encoders import ACTION_DIM, ACTOR_DIM, CRITIC_DIM
from guandan.pvguan.ppo import PPOConfig, PPOTrainer


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_net(hidden: int = 32) -> ActorCriticNet:
    torch.manual_seed(0)
    return ActorCriticNet(
        d_state_actor=ACTOR_DIM,
        d_action=ACTION_DIM,
        d_state_critic=CRITIC_DIM,
        hidden=hidden,
    )


def _make_batch(N: int = 8, K: int = 5, device: torch.device | None = None):
    """Synthetic PPOBatch with N decisions and K uniform candidates each."""
    from guandan.pvguan.buffer import PPOBatch
    dev = device or torch.device("cpu")
    torch.manual_seed(1)
    sa   = torch.randn(N, K, ACTOR_DIM)
    acts = torch.randn(N, K, ACTION_DIM)
    mask = torch.ones(N, K, dtype=torch.bool)
    sidx = torch.zeros(N, dtype=torch.long)           # always picks candidate 0
    sa_flat = sa.view(N * K, -1)
    acts_flat = acts.view(N * K, -1)

    # Use a fresh net to set consistent log_probs
    net = _make_net()
    with torch.no_grad():
        logits_flat = net.score_actions(sa_flat, acts_flat)
        logits = logits_flat.view(N, K).masked_fill(~mask, -1e9)
        lp = F.log_softmax(logits, dim=-1)
        old_lp = lp.gather(1, sidx.unsqueeze(1)).squeeze(1)
        sc = torch.randn(N, CRITIC_DIM)
        v_old = net.value(sc)

    adv = torch.randn(N)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ret = adv + v_old.detach()

    return PPOBatch(
        state_actor  = sa,
        actions      = acts,
        legal_mask   = mask,
        sampled_idx  = sidx,
        old_log_prob = old_lp.detach(),
        state_critic = sc,
        v_old        = v_old.detach(),
        returns      = ret,
        advantages   = adv,
    ), net


# ─── Ratio / clip math ───────────────────────────────────────────────────────

def test_ratio_equals_one_at_start():
    """With the same network params used for old_log_prob, ratio = 1."""
    batch, net = _make_batch()
    N, K, _ = batch.state_actor.shape

    sa_flat   = batch.state_actor.view(N * K, -1)
    acts_flat = batch.actions.view(N * K, -1)
    with torch.no_grad():
        logits = net.score_actions(sa_flat, acts_flat).view(N, K)
        logits = logits.masked_fill(~batch.legal_mask, -1e9)
        lp = F.log_softmax(logits, dim=-1)
        new_lp = lp.gather(1, batch.sampled_idx.unsqueeze(1)).squeeze(1)
        ratio = (new_lp - batch.old_log_prob).exp()

    assert torch.allclose(ratio, torch.ones(N), atol=1e-5), \
        f"Expected ratio ≈ 1.0 but got {ratio}"


def test_clip_fraction_zero_when_ratio_one():
    """If ratio = 1 exactly, clip fraction must be 0."""
    batch, net = _make_batch()
    cfg = PPOConfig(clip_eps=0.2, c_entropy=0.0, kl_init=0.0,
                    critic_warmup_iters=0, n_epochs=1, mini_batch_size=8)
    trainer = PPOTrainer(net, cfg, torch.device("cpu"))
    metrics = trainer.update(batch)
    assert metrics["clip_fraction"] == pytest.approx(0.0, abs=1e-4)


def test_clip_triggers_when_ratio_far():
    """Artificially large advantages with clipped ratio should give clip_fraction > 0."""
    batch, net = _make_batch(N=32, K=5)
    # Make old_log_prob very different from current net
    import dataclasses
    batch = dataclasses.replace(batch, old_log_prob=batch.old_log_prob + 5.0)   # ratio ≈ e^5 >> 1
    cfg = PPOConfig(clip_eps=0.2, c_entropy=0.0, kl_init=0.0,
                    critic_warmup_iters=0, n_epochs=1, mini_batch_size=32)
    trainer = PPOTrainer(net, cfg, torch.device("cpu"))
    metrics = trainer.update(batch)
    assert metrics["clip_fraction"] > 0.5


# ─── Masked softmax ──────────────────────────────────────────────────────────

def test_masked_logits_illegal_never_selected():
    """After masking, log_prob of illegal action is < -100."""
    net = _make_net()
    N, K = 4, 6
    sa   = torch.randn(N * K, ACTOR_DIM)
    acts = torch.randn(N * K, ACTION_DIM)
    with torch.no_grad():
        logits = net.score_actions(sa, acts).view(N, K)
    mask = torch.ones(N, K, dtype=torch.bool)
    mask[:, -1] = False   # last candidate is illegal for all decisions
    logits_masked = logits.masked_fill(~mask, -1e9)
    lp = F.log_softmax(logits_masked, dim=-1)
    assert (lp[:, -1] < -100).all()


def test_legal_log_probs_sum_to_one():
    """Legal action probs must sum to 1 per decision."""
    net = _make_net()
    N, K = 4, 6
    sa   = torch.randn(N * K, ACTOR_DIM)
    acts = torch.randn(N * K, ACTION_DIM)
    with torch.no_grad():
        logits = net.score_actions(sa, acts).view(N, K)
    mask = torch.ones(N, K, dtype=torch.bool)
    mask[:, 4:] = False
    logits = logits.masked_fill(~mask, -1e9)
    probs = F.softmax(logits, dim=-1)
    legal_sum = (probs * mask.float()).sum(dim=-1)
    assert torch.allclose(legal_sum, torch.ones(N), atol=1e-5)


# ─── Entropy ─────────────────────────────────────────────────────────────────

def test_entropy_positive_after_warmup():
    """Policy entropy must be > 0 with a non-degenerate distribution."""
    net = _make_net()
    N, K = 16, 8
    sa   = torch.randn(N * K, ACTOR_DIM)
    acts = torch.randn(N * K, ACTION_DIM)
    mask = torch.ones(N, K, dtype=torch.bool)
    with torch.no_grad():
        logits = net.score_actions(sa, acts).view(N, K)
        logits = logits.masked_fill(~mask, -1e9)
        lp = F.log_softmax(logits, dim=-1)
        p = lp.exp() * mask.float()
        ent = -(p * lp.clamp(min=-1e9)).sum(dim=-1).mean()
    assert ent.item() > 0.01


# ─── KL direction and schedule ───────────────────────────────────────────────

def test_kl_warm_to_theta_nonneg():
    """KL(π_warm || π_θ) must be ≥ 0 by definition."""
    from guandan.pvguan.ppo import _kl_warm_to_theta
    N, K = 8, 5
    torch.manual_seed(2)
    log_warm  = F.log_softmax(torch.randn(N, K), dim=-1)
    log_theta = F.log_softmax(torch.randn(N, K), dim=-1)
    mask = torch.ones(N, K, dtype=torch.bool)
    kl = _kl_warm_to_theta(log_warm, log_theta, mask)
    assert kl.item() >= 0.0


def test_kl_zero_when_identical():
    """KL(π || π) = 0."""
    from guandan.pvguan.ppo import _kl_warm_to_theta
    N, K = 4, 5
    logits = torch.randn(N, K)
    lp = F.log_softmax(logits, dim=-1)
    mask = torch.ones(N, K, dtype=torch.bool)
    kl = _kl_warm_to_theta(lp, lp, mask)
    assert kl.item() == pytest.approx(0.0, abs=1e-5)


def test_kl_coefficient_frozen_during_warmup():
    """c_KL must equal kl_init during the critic warmup phase."""
    net = _make_net()
    cfg = PPOConfig(kl_init=0.07, critic_warmup_iters=10, kl_decay_actor_iters=20)
    trainer = PPOTrainer(net, cfg, torch.device("cpu"))
    for _ in range(5):
        assert trainer._c_kl() == pytest.approx(0.07)
        trainer._iter += 1


def test_kl_coefficient_decays_after_unfreeze():
    """c_KL decays linearly from kl_init to 0 after actor unfreezes."""
    net = _make_net()
    cfg = PPOConfig(kl_init=0.1, critic_warmup_iters=0, kl_decay_actor_iters=10)
    trainer = PPOTrainer(net, cfg, torch.device("cpu"))
    # Iter 0: first call to _c_kl sets _actor_unfrozen_at
    c0 = trainer._c_kl()
    assert c0 == pytest.approx(0.1, abs=1e-6)
    # Manually advance iters
    trainer._actor_unfrozen_at = 0
    trainer._iter = 5
    c5 = trainer._c_kl()
    assert c5 == pytest.approx(0.05, abs=1e-6)
    trainer._iter = 10
    c10 = trainer._c_kl()
    assert c10 == pytest.approx(0.0, abs=1e-6)


def test_kl_zero_after_decay_completes():
    """After full decay, c_KL stays at 0."""
    net = _make_net()
    cfg = PPOConfig(kl_init=0.1, critic_warmup_iters=0, kl_decay_actor_iters=5)
    trainer = PPOTrainer(net, cfg, torch.device("cpu"))
    trainer._actor_unfrozen_at = 0
    trainer._iter = 100
    assert trainer._c_kl() == pytest.approx(0.0, abs=1e-9)


# ─── Gradient flow ───────────────────────────────────────────────────────────

def _actor_params_changed(net_before: ActorCriticNet, net_after: ActorCriticNet) -> bool:
    for (n, p_b), (_, p_a) in zip(
        net_before.actor_head.named_parameters(),
        net_after.actor_head.named_parameters(),
    ):
        if not torch.equal(p_b, p_a):
            return True
    return False


def _critic_params_changed(net_before: ActorCriticNet, net_after: ActorCriticNet) -> bool:
    for (_, p_b), (_, p_a) in zip(
        net_before.critic_head.named_parameters(),
        net_after.critic_head.named_parameters(),
    ):
        if not torch.equal(p_b, p_a):
            return True
    return False


def test_actor_frozen_during_critic_warmup():
    """Actor parameters must not change during critic_warmup_iters."""
    net = _make_net()
    import copy
    net_before = copy.deepcopy(net)

    batch, _ = _make_batch()
    cfg = PPOConfig(
        critic_warmup_iters=5,
        kl_init=0.0,
        c_entropy=0.0,
        n_epochs=2,
        mini_batch_size=8,
    )
    trainer = PPOTrainer(net, cfg, torch.device("cpu"))
    # Run one update while actor is frozen
    assert trainer.actor_frozen
    trainer.update(batch)

    assert not _actor_params_changed(net_before, net), \
        "Actor parameters changed during critic warm-up (should be frozen)"


def test_critic_trains_during_warmup():
    """Critic must update even while actor is frozen."""
    net = _make_net()
    import copy
    net_before = copy.deepcopy(net)

    batch, _ = _make_batch()
    cfg = PPOConfig(
        critic_warmup_iters=5,
        kl_init=0.0,
        c_entropy=0.0,
        n_epochs=2,
        mini_batch_size=8,
    )
    trainer = PPOTrainer(net, cfg, torch.device("cpu"))
    trainer.update(batch)

    assert _critic_params_changed(net_before, net), \
        "Critic parameters did not change during warm-up"


def test_actor_trains_after_warmup():
    """Actor must update after critic warm-up phase ends."""
    net = _make_net()
    import copy

    batch, _ = _make_batch()
    cfg = PPOConfig(
        critic_warmup_iters=0,   # no warmup — actor trains immediately
        kl_init=0.0,
        c_entropy=0.0,
        n_epochs=2,
        mini_batch_size=8,
    )
    trainer = PPOTrainer(net, cfg, torch.device("cpu"))
    net_before = copy.deepcopy(net)
    trainer.update(batch)

    assert _actor_params_changed(net_before, net), \
        "Actor parameters did not change after critic warm-up"


# ─── actor_frozen flag ───────────────────────────────────────────────────────

def test_actor_frozen_flag_transitions():
    """actor_frozen=True for iters < warmup, False after."""
    net = _make_net()
    cfg = PPOConfig(critic_warmup_iters=3)
    trainer = PPOTrainer(net, cfg, torch.device("cpu"))
    assert trainer.actor_frozen is True
    trainer._iter = 2
    assert trainer.actor_frozen is True
    trainer._iter = 3
    assert trainer.actor_frozen is False


# ─── Update return schema ─────────────────────────────────────────────────────

def test_update_returns_required_keys():
    """update() must return all expected metric keys."""
    net = _make_net()
    batch, _ = _make_batch()
    cfg = PPOConfig(critic_warmup_iters=0, kl_init=0.05, n_epochs=1, mini_batch_size=8)
    trainer = PPOTrainer(net, cfg, torch.device("cpu"))
    metrics = trainer.update(batch)
    required = {
        "loss_total", "loss_policy", "loss_value",
        "loss_entropy", "loss_kl_warmstart",
        "clip_fraction", "approx_kl",
        "explained_variance_current",
        "c_kl_current", "actor_frozen",
    }
    assert required <= metrics.keys()
