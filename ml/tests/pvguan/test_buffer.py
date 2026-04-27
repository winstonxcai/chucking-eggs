"""Per-player GAE tests for pvguan/buffer.py.

Three cases:
  A: full participation — each player acts every step until hand end
  B: early finisher — player 0 stops after 2 decisions; hand continues
  C: 4-player interleaving with cross-player bootstrap correctness
"""

from __future__ import annotations

import numpy as np
import pytest

from guandan.pvguan.buffer import Decision, PlayerTrack, RolloutBuffer


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _dummy_state_actor(K: int = 2) -> np.ndarray:
    return np.zeros((K, 764), dtype=np.float32)


def _dummy_actions(K: int = 2) -> np.ndarray:
    return np.zeros((K, 198), dtype=np.float32)


def _dummy_mask(K: int = 2) -> np.ndarray:
    m = np.zeros(K, dtype=bool)
    m[0] = True
    return m


def _dummy_state_critic() -> np.ndarray:
    return np.zeros(875, dtype=np.float32)


def _decision(v_old: float, K: int = 2) -> Decision:
    return Decision(
        state_actor=_dummy_state_actor(K),
        actions=_dummy_actions(K),
        legal_mask=_dummy_mask(K),
        sampled_idx=0,
        log_prob=-1.0,
        state_critic=_dummy_state_critic(),
        v_old=v_old,
    )


def _gae(rewards, v_olds, gamma=1.0, lam=0.95):
    T = len(rewards)
    advantages = np.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        v_next = v_olds[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * v_next - v_olds[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    return advantages


# ─── Case A: full participation ───────────────────────────────────────────────

def test_gae_full_participation():
    gamma, lam = 1.0, 0.95
    buf = RolloutBuffer(gamma=gamma, lam=lam)

    # Player 0: 4 decisions, terminal reward = 0.333
    reward = 1.0 / 3
    v_olds = [0.1, 0.2, 0.15, 0.05]

    track = PlayerTrack(player=0)
    for v in v_olds:
        track.add(_decision(v))
    track.finalize(reward)
    buf.add_track(track)

    batch = buf.compute_batch()

    rewards = np.array([0.0, 0.0, 0.0, reward])
    expected_adv = _gae(rewards, v_olds, gamma, lam)
    expected_ret = expected_adv + np.array(v_olds)

    adv_norm = (expected_adv - expected_adv.mean()) / (expected_adv.std() + 1e-8)

    actual_adv = batch.advantages.numpy()
    actual_ret = batch.returns.numpy()

    assert np.allclose(actual_adv, adv_norm, atol=1e-5), \
        f"Advantages mismatch: {actual_adv} vs {adv_norm}"
    assert np.allclose(actual_ret, expected_ret, atol=1e-5), \
        f"Returns mismatch: {actual_ret} vs {expected_ret}"


# ─── Case B: early finisher ───────────────────────────────────────────────────

def test_gae_early_finisher():
    """Player 0 acts 2 times then goes out; hand continues for others."""
    gamma, lam = 1.0, 0.95
    buf = RolloutBuffer(gamma=gamma, lam=lam)

    reward = -1.0 / 3
    v_olds_p0 = [0.3, 0.1]

    track0 = PlayerTrack(player=0)
    for v in v_olds_p0:
        track0.add(_decision(v))
    track0.finalize(reward)

    # Player 2 continues for 5 more steps
    v_olds_p2 = [0.2, 0.1, 0.05, 0.15, 0.0]
    track2 = PlayerTrack(player=2)
    for v in v_olds_p2:
        track2.add(_decision(v))
    track2.finalize(reward)

    buf.add_track(track0)
    buf.add_track(track2)

    batch = buf.compute_batch()

    # Player 0: 2 steps, V_next(t=1) = 0 (terminal)
    rewards_p0 = np.array([0.0, reward])
    exp_adv_p0 = _gae(rewards_p0, v_olds_p0, gamma, lam)

    # Player 2: 5 steps
    rewards_p2 = np.array([0.0, 0.0, 0.0, 0.0, reward])
    exp_adv_p2 = _gae(rewards_p2, v_olds_p2, gamma, lam)

    all_adv = np.concatenate([exp_adv_p0, exp_adv_p2])
    all_adv_norm = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)

    actual_adv = batch.advantages.numpy()
    assert np.allclose(actual_adv, all_adv_norm, atol=1e-5), \
        f"Early finisher advantages mismatch"

    # Player 0 track has 2 decisions
    assert batch.advantages.shape[0] == 7


# ─── Case C: interleaved 4-player trajectories ───────────────────────────────

def test_gae_interleaved_does_not_cross_players():
    """P0's GAE bootstraps from P0's next state only, not P1/P2/P3."""
    gamma, lam = 1.0, 0.95
    buf = RolloutBuffer(gamma=gamma, lam=lam)

    reward = 1.0 / 3
    # Each player has exactly 3 decisions
    v_matrix = {
        0: [0.10, 0.20, 0.15],
        1: [0.05, 0.08, 0.03],
        2: [0.12, 0.18, 0.10],
        3: [0.07, 0.11, 0.06],
    }
    for p, v_olds in v_matrix.items():
        track = PlayerTrack(player=p)
        for v in v_olds:
            track.add(_decision(v))
        track.finalize(reward)
        buf.add_track(track)

    batch = buf.compute_batch()

    # Verify P0's returns are independent of P1/P2/P3
    # Expected P0 returns
    rewards_p0 = np.array([0.0, 0.0, reward])
    exp_adv_p0 = _gae(rewards_p0, v_matrix[0], gamma, lam)
    exp_ret_p0 = exp_adv_p0 + np.array(v_matrix[0])

    all_rets = batch.returns.numpy()
    # P0 is first 3 entries (tracks added in order 0,1,2,3 each with 3 decisions)
    actual_ret_p0 = all_rets[:3]
    assert np.allclose(actual_ret_p0, exp_ret_p0, atol=1e-5), \
        f"P0 returns: {actual_ret_p0} vs expected {exp_ret_p0}"


# ─── Frozen V_old ─────────────────────────────────────────────────────────────

def test_v_old_frozen_in_batch():
    """v_old in batch matches the rollout-time values (not recomputed)."""
    buf = RolloutBuffer()
    v_olds = [0.1, 0.2, 0.3]
    track = PlayerTrack(player=0)
    for v in v_olds:
        track.add(_decision(v))
    track.finalize(1.0)
    buf.add_track(track)

    batch = buf.compute_batch()
    assert np.allclose(batch.v_old.numpy(), v_olds, atol=1e-6)


# ─── Buffer clear ─────────────────────────────────────────────────────────────

def test_buffer_clear():
    buf = RolloutBuffer()
    track = PlayerTrack(player=0)
    track.add(_decision(0.1))
    track.finalize(0.5)
    buf.add_track(track)
    assert buf.size() == 1
    buf.clear()
    assert buf.size() == 0
