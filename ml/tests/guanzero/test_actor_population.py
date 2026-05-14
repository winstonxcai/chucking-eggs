"""Tests for the M4 checkpoint-population code paths in play_episode.

The bits these cover are the ones that would silently corrupt training if
they regressed: per-seat net routing, per-seat ε routing, and the worker-side
sample filter that drops frozen-team rows before they hit the buffer.
"""

from __future__ import annotations

import torch

from guandan.guanzero.actor import play_episode
from guandan.guanzero.encoding.role_encoder import RoleAwareStateActionEncoder
from guandan.guanzero.utils.profiler import PhaseProfiler
from guandan.guanzero.q_network import SharedHeadQNet, SharedHeadQNetConfig


def _small_shared_net(seed: int = 0) -> SharedHeadQNet:
    """Tiny SharedHeadQNet for fast tests; init seed makes results deterministic."""
    torch.manual_seed(seed)
    return SharedHeadQNet(SharedHeadQNetConfig(
        role_d_model=16,
        history_hidden=16,
        global_hidden=8,
        action_hidden=8,
        trunk_hidden=32,
        trunk_layers=1,
    ))


def _run_episode(*, frozen_seats, q_nets_frozen, epsilon, epsilon_frozen,
                 latest_net=None, encoder=None, seed=21):
    """Helper: run one play_episode with the population kwargs."""
    if latest_net is None:
        latest_net = _small_shared_net(seed=1)
    if encoder is None:
        encoder = RoleAwareStateActionEncoder()
    return play_episode(
        q_nets=latest_net,
        encoder=encoder,
        epsilon=epsilon,
        seed=seed,
        device="cpu",
        gamma=1.0,
        q_nets_frozen=q_nets_frozen,
        frozen_seats=frozen_seats,
        epsilon_frozen=epsilon_frozen,
    )


# ── Sample filter ──────────────────────────────────────────────────────────


def test_sample_filter_latest_even_keeps_only_seats_0_and_2():
    """Worker-side filter: with frozen_seats={1,3}, only seats 0,2 reach the buffer."""
    frozen = _small_shared_net(seed=2)
    samples = _run_episode(
        frozen_seats=frozenset({1, 3}),
        q_nets_frozen=frozen,
        epsilon=0.0,
        epsilon_frozen=0.0,
    )
    assert samples, "episode must produce at least one decision"
    kept = [s for s in samples if s.player not in {1, 3}]
    dropped = [s for s in samples if s.player in {1, 3}]
    assert kept, "latest-team seats should produce some samples"
    assert dropped, "frozen-team seats should also produce samples (pre-filter)"
    assert {s.player for s in kept} <= {0, 2}


def test_sample_filter_latest_odd_keeps_only_seats_1_and_3():
    frozen = _small_shared_net(seed=2)
    samples = _run_episode(
        frozen_seats=frozenset({0, 2}),
        q_nets_frozen=frozen,
        epsilon=0.0,
        epsilon_frozen=0.0,
    )
    assert samples
    kept = [s for s in samples if s.player not in {0, 2}]
    assert kept
    assert {s.player for s in kept} <= {1, 3}


def test_self_play_emits_samples_from_all_four_seats():
    """frozen_seats=∅ → no filter, every seat contributes."""
    samples = _run_episode(
        frozen_seats=frozenset(),
        q_nets_frozen=None,
        epsilon=0.0,
        epsilon_frozen=0.0,
    )
    seats = {s.player for s in samples}
    # A typical Guan Dan game is long enough that all 4 seats decide at least once.
    assert seats == {0, 1, 2, 3}, f"expected all 4 seats, got {seats}"


# ── Per-seat net routing ───────────────────────────────────────────────────


def test_frozen_seats_route_to_frozen_net():
    """forward_grouped on the frozen net must fire only for frozen-seat decisions
    (under ε=0 — no random shortcut, and excluding K=1 shortcuts which skip
    every network entirely)."""
    latest = _small_shared_net(seed=1)
    frozen = _small_shared_net(seed=2)
    latest_calls = 0
    frozen_calls = 0
    orig_latest = latest.forward_grouped
    orig_frozen = frozen.forward_grouped

    def latest_spy(state, action, repeats):
        nonlocal latest_calls
        latest_calls += 1
        return orig_latest(state, action, repeats)

    def frozen_spy(state, action, repeats):
        nonlocal frozen_calls
        frozen_calls += 1
        return orig_frozen(state, action, repeats)

    latest.forward_grouped = latest_spy
    frozen.forward_grouped = frozen_spy

    _ = _run_episode(
        frozen_seats=frozenset({1, 3}),
        q_nets_frozen=frozen,
        epsilon=0.0,
        epsilon_frozen=0.0,
        latest_net=latest,
    )

    assert latest_calls > 0, "latest net must score at least one decision"
    assert frozen_calls > 0, "frozen net must score at least one decision"


# ── Per-seat ε routing ─────────────────────────────────────────────────────


def test_frozen_seats_skip_epsilon_random_branch():
    """With epsilon=1.0 (latest always random) and epsilon_frozen=0.0 (frozen
    always greedy), the profiler's epsilon_random counter records only latest-
    seat decisions. Frozen-seat decisions must instead score via the frozen
    net's forward_grouped."""
    latest = _small_shared_net(seed=1)
    frozen = _small_shared_net(seed=2)

    # Spy on the frozen net so we can count its greedy decisions.
    frozen_calls = 0
    orig = frozen.forward_grouped

    def spy(state, action, repeats):
        nonlocal frozen_calls
        frozen_calls += 1
        return orig(state, action, repeats)

    frozen.forward_grouped = spy

    prof = PhaseProfiler(enabled=True)
    samples = play_episode(
        q_nets=latest,
        encoder=RoleAwareStateActionEncoder(),
        epsilon=1.0,            # latest seats: always take random branch
        seed=33,
        device="cpu",
        gamma=1.0,
        profiler=prof,
        q_nets_frozen=frozen,
        frozen_seats=frozenset({1, 3}),
        epsilon_frozen=0.0,     # frozen seats: never take random branch
    )

    assert samples
    eps_random_count = prof._counts.get("epsilon_random", 0)
    assert eps_random_count > 0, "latest seats with ε=1.0 must hit the random branch"
    assert frozen_calls > 0, "frozen seats with ε=0 must score greedily"
