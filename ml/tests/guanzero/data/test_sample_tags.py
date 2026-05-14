"""Per-sample diagnostic tag helpers + end-to-end propagation through play_episode."""

from __future__ import annotations

import math

import numpy as np

from guandan.cards import ComboType
from guandan.guanzero.actor import _phase, _phase_with_out, _trick_role, play_episode
from guandan.guanzero.model.encoding.role_encoder import RoleAwareStateActionEncoder
from guandan.guanzero.model.q_network import SharedHeadQNet, SharedHeadQNetConfig
from guandan.guanzero.data.sample_tags import (
    ACTION_CLASS_BOMB,
    ACTION_CLASS_JOKER_BOMB,
    ACTION_CLASS_LOOKUP,
    ACTION_CLASS_MULTI,
    ACTION_CLASS_PAIR_TRIPLE,
    ACTION_CLASS_PASS,
    ACTION_CLASS_SINGLE,
    collapse_action_type,
    k_bucket,
    k_bucket_array,
    q_gap_bucket,
    q_gap_bucket_array,
    reward_bucket,
    reward_bucket_array,
)


def test_phase_helper_bins():
    assert _phase(27) == 0
    assert _phase(20) == 0
    assert _phase(19) == 1
    assert _phase(10) == 1
    assert _phase(9) == 2
    assert _phase(0) == 2


def test_phase_with_out_returns_3_when_out():
    class FakeEnv:
        is_out = [False, True, False, False]
        hands = [list(range(15)), [], list(range(5)), list(range(25))]

    env = FakeEnv()
    assert _phase_with_out(env, 0) == 1  # 15 cards → midgame
    assert _phase_with_out(env, 1) == 3  # out
    assert _phase_with_out(env, 2) == 2  # 5 cards → endgame
    assert _phase_with_out(env, 3) == 0  # 25 cards → opening


def test_trick_role_branches():
    class FakeEnv:
        def __init__(self, leading: bool, partner_out: bool):
            self._leading = leading
            self.is_out = [False, False, partner_out, False]

        def is_leading(self):
            return self._leading

    assert _trick_role(FakeEnv(leading=True, partner_out=False), 2) == 0
    assert _trick_role(FakeEnv(leading=False, partner_out=False), 2) == 1
    assert _trick_role(FakeEnv(leading=False, partner_out=True), 2) == 2


def test_collapse_action_type_buckets():
    assert collapse_action_type(int(ComboType.PASS)) == ACTION_CLASS_PASS
    assert collapse_action_type(int(ComboType.SINGLE)) == ACTION_CLASS_SINGLE
    assert collapse_action_type(int(ComboType.PAIR)) == ACTION_CLASS_PAIR_TRIPLE
    assert collapse_action_type(int(ComboType.TRIPLE)) == ACTION_CLASS_PAIR_TRIPLE
    for ct in (ComboType.FULL_HOUSE, ComboType.STRAIGHT, ComboType.TUBE, ComboType.PLATE):
        assert collapse_action_type(int(ct)) == ACTION_CLASS_MULTI
    for ct in (
        ComboType.BOMB_4, ComboType.BOMB_5, ComboType.STRAIGHT_FLUSH,
        ComboType.BOMB_6, ComboType.BOMB_7, ComboType.BOMB_8,
        ComboType.BOMB_9, ComboType.BOMB_10,
    ):
        assert collapse_action_type(int(ct)) == ACTION_CLASS_BOMB
    assert collapse_action_type(int(ComboType.BOMB_JOKER)) == ACTION_CLASS_JOKER_BOMB


def test_action_class_lookup_table_matches_collapse():
    for ct in ComboType:
        assert ACTION_CLASS_LOOKUP[int(ct)] == collapse_action_type(int(ct))


def test_k_bucket_boundaries():
    assert k_bucket(1) == 0
    assert k_bucket(2) == 1
    assert k_bucket(5) == 1
    assert k_bucket(6) == 2
    assert k_bucket(20) == 2
    assert k_bucket(21) == 3
    arr = np.array([1, 2, 5, 6, 20, 21, 100])
    assert k_bucket_array(arr).tolist() == [0, 1, 1, 2, 2, 3, 3]


def test_q_gap_bucket_boundaries():
    assert q_gap_bucket(float("nan")) == 0
    assert q_gap_bucket(0.0) == 1
    assert q_gap_bucket(0.049) == 1
    assert q_gap_bucket(0.05) == 2
    assert q_gap_bucket(0.19) == 2
    assert q_gap_bucket(0.20) == 3
    assert q_gap_bucket(1.0) == 3
    arr = np.array([float("nan"), 0.0, 0.05, 0.19, 0.20, 1.0])
    assert q_gap_bucket_array(arr).tolist() == [0, 1, 2, 2, 3, 3]


def test_reward_bucket_boundaries():
    assert reward_bucket(-3.0) == 0
    assert reward_bucket(-2.0) == 0
    assert reward_bucket(-1.0) == 1
    assert reward_bucket(0.0) == 2
    assert reward_bucket(1.0) == 2
    assert reward_bucket(2.0) == 3
    assert reward_bucket(3.0) == 3
    arr = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    assert reward_bucket_array(arr).tolist() == [0, 0, 1, 2, 2, 3, 3]


def _small_shared_net() -> SharedHeadQNet:
    return SharedHeadQNet(SharedHeadQNetConfig(
        role_d_model=16,
        history_hidden=16,
        global_hidden=8,
        action_hidden=8,
        trunk_hidden=32,
        trunk_layers=1,
    ))


def test_play_episode_tags_propagate():
    """Run a real episode; verify the 13 tag fields are populated on every sample."""
    samples = play_episode(
        q_nets=_small_shared_net(),
        encoder=RoleAwareStateActionEncoder(),
        epsilon=1.0,  # random play → fast termination, exercises epsilon branch
        seed=99,
        device="cpu",
        gamma=1.0,
        episode_mode=2,
        opponent_id=3,
        latest_team=1,
    )
    assert samples

    for s in samples:
        assert s.episode_mode == 2
        assert s.opponent_id == 3
        assert s.latest_team == 1
        assert s.terminal_reward in (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)
        assert s.phase_self in (0, 1, 2)
        assert s.trick_role in (0, 1, 2)
        assert s.phase_partner in (0, 1, 2, 3)
        assert s.action_type in [int(t) for t in ComboType]
        assert s.is_pass in (0, 1)
        assert s.is_bomb in (0, 1)
        assert s.num_legal_actions >= 1
        # With epsilon=1.0 every non-K=1 decision is epsilon-random, so q_gap is NaN.
        assert math.isnan(s.q_gap)
        assert s.chosen_by_epsilon in (0, 1)
        # Derived flags must match action_type.
        assert s.is_pass == int(s.action_type == 0)
        assert s.is_bomb == int(s.action_type >= int(ComboType.BOMB_4))
