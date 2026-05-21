from __future__ import annotations

import math
import random

import torch

import guandan.dart.runtime.actor.rollout as actor_mod
from guandan.agents import make_agent
from guandan.cards import Card, ComboType, Rank, Suit
from guandan.combos import Combo
from guandan.game import GuanDanEnv
from guandan.dart.data.returns import EpisodeTags
from guandan.dart.data.sample_tags import (
    EPISODE_MODE_SELF_PLAY,
    EPISODE_MODE_VS_CHECKPOINT,
    EPISODE_MODE_VS_HARD_BOT,
    OPPONENT_CHECKPOINT_BASE,
    OPPONENT_STRATEGIC,
)
from guandan.dart.runtime.actor import (
    LaneConfig,
    SeatPolicy,
    argmax_q,
    all_latest_seats,
    play_episode,
    play_episodes_batched,
    select_legal,
)
from guandan.dart.data.buffer import collate_base_encoded, collate_role_encoded
from guandan.dart.config import QNetConfig
from guandan.dart.model.encoding.base_encoder import StateActionEncoder
from guandan.dart.model.encoding.role_encoder import ROLE_ENCODE_CHANNEL_KEYS, RoleAwareStateActionEncoder
from guandan.dart.model.q_network import (
    DartQNet,
    DartQNetConfig,
    init_guanzero_nets,
)
from guandan.dart.data.returns import TrainSample


def _with_seats(base, replacements):
    seats = list(base)
    for seat, policy in replacements.items():
        seats[seat] = policy
    return tuple(seats)


def _script_combo(key: int) -> Combo:
    return Combo(ComboType.SINGLE, key, [])


class _ScriptedLaneEnv:
    def __init__(self, legal_by_step: list[list[Combo]]) -> None:
        self.legal_by_step = legal_by_step
        self.step_idx = 0
        self.stepped_actions: list[Combo] = []
        self.current_player = 0
        self.current_trick = None
        self.trick_winner = None
        self.hands = [set(range(5)), set(range(5)), set(range(5)), set(range(5))]
        self.is_out = [False, False, False, False]
        self.done = False

    def step(self, combo: Combo):
        self.stepped_actions.append(combo)
        self.step_idx += 1
        if self.step_idx >= len(self.legal_by_step):
            self.done = True
        return self.current_player, self.done

    def is_leading(self) -> bool:
        return self.current_trick is None

    def get_rewards(self) -> dict[int, float]:
        return {0: 3.0, 1: -3.0, 2: 3.0, 3: -3.0}


class _CountingScriptEncoder:
    def __init__(self) -> None:
        self.encode_all_calls = 0
        self.encode_one_calls = 0

    def encode_all(self, env, player, legal_moves):
        self.encode_all_calls += 1
        return [{"choice": i} for i, _move in enumerate(legal_moves)]

    def encode_one(self, env, player, action, legal_moves):
        self.encode_one_calls += 1
        return {"selected": legal_moves.index(action)}


def _scripted_q_nets() -> dict[int, object]:
    return {p: object() for p in range(4)}


def test_one_episode_returns_team_signed_samples():
    encoder = StateActionEncoder()
    q_nets = init_guanzero_nets(QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2))
    samples = play_episode(
        q_nets=q_nets,
        encoder=encoder,
        seats=all_latest_seats(1.0),  # pure random play for fastest termination
        seed=7,
        device="cpu",
        gamma=1.0,
    )
    assert len(samples) > 0
    assert all(isinstance(s, TrainSample) for s in samples)
    # Team-signed: per-player returns should be ±1, summing to zero across teams
    by_player = {p: [] for p in range(4)}
    for s in samples:
        by_player[s.player].append(s.mc_return)
    for p, vals in by_player.items():
        if vals:
            assert all(v == vals[0] for v in vals)  # constant across player's traj (gamma=1)
            assert vals[0] in (1.0, -1.0, 2/3, -2/3, 1/3, -1/3)
    # Teammates share sign
    if by_player[0] and by_player[2]:
        assert (by_player[0][0] > 0) == (by_player[2][0] > 0)
    if by_player[1] and by_player[3]:
        assert (by_player[1][0] > 0) == (by_player[3][0] > 0)


class _FakeEnv:
    def __init__(self, legal, leading: bool):
        self._legal = legal
        self._leading = leading

    def legal_moves(self, player: int):
        assert player == 0
        return list(self._legal)

    def is_leading(self) -> bool:
        return self._leading


def test_select_legal_dedups_and_adds_pass_when_responding():
    pair_a = Combo(
        ComboType.PAIR,
        Rank.ACE,
        [Card(Rank.ACE, Suit.SPADE, 0), Card(Rank.ACE, Suit.HEART, 0)],
    )
    pair_b = Combo(
        ComboType.PAIR,
        Rank.ACE,
        [Card(Rank.ACE, Suit.DIAMOND, 0), Card(Rank.ACE, Suit.CLUB, 0)],
    )

    legal = select_legal(_FakeEnv([pair_a, pair_b], leading=False), 0)
    assert len(legal) == 2
    assert legal[0].type == pair_a.type and legal[0].key == pair_a.key
    assert legal[-1].type == ComboType.PASS


def test_select_legal_does_not_add_pass_when_leading():
    single = Combo(ComboType.SINGLE, Rank.ACE, [Card(Rank.ACE, Suit.SPADE, 0)])

    legal = select_legal(_FakeEnv([single], leading=True), 0)
    assert len(legal) == 1
    assert legal[0].type == single.type and legal[0].key == single.key


def test_argmax_q_matches_grouped_forward():
    torch.manual_seed(11)
    encoder = StateActionEncoder()
    q_nets = init_guanzero_nets(QNetConfig(
        hidden_lstm=16,
        hidden_mlp=32,
        n_mlp_layers=2,
        history_encoder="transformer",
        transformer_nhead=4,
        transformer_layers=1,
        transformer_ff_dim=64,
    ))
    env = GuanDanEnv()
    env.reset(seed=5)
    p = env.current_player
    legal = select_legal(env, p)[:6]
    encoded = encoder.encode_all(env, p, legal)

    with torch.no_grad():
        state_batch, action_batch, repeats = collate_base_encoded([encoded])
        expected = int(
            q_nets[p].forward_grouped(state_batch, action_batch, repeats).argmax().item()
        )
    idx, _ = argmax_q(q_nets[p], encoded, torch.device("cpu"))
    assert idx == expected


def _small_dart_net() -> DartQNet:
    return DartQNet(DartQNetConfig(
        role_d_model=16,
        history_hidden=16,
        global_hidden=8,
        action_hidden=8,
        trunk_hidden=32,
        trunk_layers=1,
    ))


def test_argmax_q_role_matches_grouped_forward():
    torch.manual_seed(12)
    encoder = RoleAwareStateActionEncoder()
    net = _small_dart_net()
    env = GuanDanEnv()
    env.reset(seed=6)
    p = env.current_player
    legal = select_legal(env, p)[:6]
    encoded = encoder.encode_all(env, p, legal)

    with torch.no_grad():
        state_batch, action_batch, repeats = collate_role_encoded([encoded])
        expected = int(net.forward_grouped(state_batch, action_batch, repeats).argmax().item())
    idx, _ = argmax_q(net, encoded, torch.device("cpu"), role_encoded=True)
    assert idx == expected


def test_role_episode_returns_samples_with_m3_keys():
    samples = play_episode(
        q_nets=_small_dart_net(),
        encoder=RoleAwareStateActionEncoder(),
        seats=all_latest_seats(1.0),
        seed=17,
        device="cpu",
        gamma=1.0,
    )

    assert samples
    assert all(isinstance(s, TrainSample) for s in samples)
    assert set(samples[0].encoded) == set(ROLE_ENCODE_CHANNEL_KEYS)


def test_batched_role_episode_returns_trick_head_samples():
    net = DartQNet(DartQNetConfig(
        role_d_model=16,
        history_hidden=16,
        global_hidden=8,
        action_hidden=8,
        trunk_hidden=32,
        trunk_layers=1,
    ))
    samples_per_lane = play_episodes_batched(
        q_nets=net,
        encoder=RoleAwareStateActionEncoder(),
        lanes=[
            LaneConfig(seed=21, seats=all_latest_seats(0.0)),
            LaneConfig(seed=22, seats=all_latest_seats(0.0)),
        ],
        device="cpu",
        gamma=1.0,
    )
    samples = [s for lane_samples in samples_per_lane for s in lane_samples]

    assert samples
    assert all(isinstance(s, TrainSample) for s in samples)
    assert "trick_head_id" in samples[0].encoded
    assert {s.episode_mode for s in samples} == {0}


def test_batched_role_episode_supports_heterogeneous_lanes():
    latest = _small_dart_net()
    frozen = _small_dart_net()
    lanes = [
        LaneConfig(
            seed=31,
            seats=all_latest_seats(0.0),
            tags=EpisodeTags(mode=EPISODE_MODE_SELF_PLAY),
        ),
        LaneConfig(
            seed=32,
            seats=_with_seats(
                all_latest_seats(0.0),
                {0: SeatPolicy.frozen(frozen, 0.0), 2: SeatPolicy.frozen(frozen, 0.0)},
            ),
            tags=EpisodeTags(
                mode=EPISODE_MODE_VS_CHECKPOINT,
                opponent_id=OPPONENT_CHECKPOINT_BASE,
                latest_team=1,
            ),
        ),
        LaneConfig(
            seed=33,
            seats=_with_seats(
                all_latest_seats(0.0),
                {1: SeatPolicy.hard_bot(make_agent("strategic")),
                 3: SeatPolicy.hard_bot(make_agent("strategic"))},
            ),
            tags=EpisodeTags(
                mode=EPISODE_MODE_VS_HARD_BOT,
                opponent_id=OPPONENT_STRATEGIC,
                latest_team=0,
            ),
        ),
    ]

    samples_per_lane = play_episodes_batched(
        q_nets=latest,
        encoder=RoleAwareStateActionEncoder(),
        lanes=lanes,
        device="cpu",
        gamma=1.0,
    )

    assert [bool(samples) for samples in samples_per_lane] == [True, True, True]
    assert {s.player for s in samples_per_lane[1]} <= {1, 3}
    assert {s.player for s in samples_per_lane[2]} <= {0, 2}
    assert {s.episode_mode for s in samples_per_lane[0]} == {EPISODE_MODE_SELF_PLAY}
    assert {s.episode_mode for s in samples_per_lane[1]} == {EPISODE_MODE_VS_CHECKPOINT}
    assert {s.opponent_id for s in samples_per_lane[1]} == {OPPONENT_CHECKPOINT_BASE}
    assert {s.episode_mode for s in samples_per_lane[2]} == {EPISODE_MODE_VS_HARD_BOT}
    assert {s.opponent_id for s in samples_per_lane[2]} == {OPPONENT_STRATEGIC}


def test_batched_role_episode_groups_same_frozen_net(monkeypatch):
    latest = _small_dart_net()
    frozen_a = _small_dart_net()
    frozen_b = _small_dart_net()
    calls: list[tuple[int, int]] = []

    def fake_argmax_q_batched(net, encoded_groups, device, **kwargs):
        calls.append((id(net), len(encoded_groups)))
        return [(0, float("nan")) for _ in encoded_groups]

    monkeypatch.setattr(actor_mod, "argmax_q_batched", fake_argmax_q_batched)

    play_episodes_batched(
        q_nets=latest,
        encoder=RoleAwareStateActionEncoder(),
        lanes=[
            LaneConfig(
                seed=41,
                seats=_with_seats(all_latest_seats(0.0), {0: SeatPolicy.frozen(frozen_a, 0.0)}),
            ),
            LaneConfig(
                seed=42,
                seats=_with_seats(all_latest_seats(0.0), {0: SeatPolicy.frozen(frozen_a, 0.0)}),
            ),
            LaneConfig(
                seed=43,
                seats=_with_seats(all_latest_seats(0.0), {0: SeatPolicy.frozen(frozen_b, 0.0)}),
            ),
        ],
        device="cpu",
        gamma=1.0,
    )

    assert (id(frozen_a), 2) in calls
    assert (id(frozen_b), 1) in calls
    assert any(net_id == id(latest) for net_id, _ in calls)


def test_batched_rollout_drains_lanes_until_q_needed(monkeypatch):
    forced = [_script_combo(Rank.THREE)]
    q_options_a = [_script_combo(Rank.FOUR), _script_combo(Rank.FIVE)]
    q_options_b = [_script_combo(Rank.SIX), _script_combo(Rank.SEVEN)]
    scripts = {
        101: [forced, q_options_a],
        102: [q_options_b],
    }
    envs: list[_ScriptedLaneEnv] = []

    def fake_env(seed=None):
        env = _ScriptedLaneEnv(scripts[seed])
        envs.append(env)
        return env

    def fake_select_legal(env, player):
        return env.legal_by_step[env.step_idx]

    q_forward_widths: list[int] = []

    def fake_argmax_q_batched(net, encoded_groups, device, **kwargs):
        q_forward_widths.append(len(encoded_groups))
        return [(0, float("nan")) for _ in encoded_groups]

    monkeypatch.setattr(actor_mod, "GuanDanEnv", fake_env)
    monkeypatch.setattr(actor_mod, "select_legal", fake_select_legal)
    monkeypatch.setattr(actor_mod, "argmax_q_batched", fake_argmax_q_batched)

    samples_per_lane = play_episodes_batched(
        q_nets=_scripted_q_nets(),
        encoder=_CountingScriptEncoder(),
        lanes=[
            LaneConfig(seed=101, seats=all_latest_seats(0.0)),
            LaneConfig(seed=102, seats=all_latest_seats(0.0)),
        ],
        device="cpu",
        gamma=1.0,
        record_forced_k1_samples=False,
    )

    assert q_forward_widths == [2]
    assert [len(samples) for samples in samples_per_lane] == [1, 1]
    assert [len(env.stepped_actions) for env in envs] == [2, 1]


def test_forced_k1_recording_flag_only_controls_sample_emission(monkeypatch):
    forced = [_script_combo(Rank.THREE)]
    envs: list[_ScriptedLaneEnv] = []

    def fake_env(seed=None):
        env = _ScriptedLaneEnv({201: [forced], 202: [forced]}[seed])
        envs.append(env)
        return env

    def fake_select_legal(env, player):
        return env.legal_by_step[env.step_idx]

    monkeypatch.setattr(actor_mod, "GuanDanEnv", fake_env)
    monkeypatch.setattr(actor_mod, "select_legal", fake_select_legal)

    skip_encoder = _CountingScriptEncoder()
    skipped = play_episodes_batched(
        q_nets=_scripted_q_nets(),
        encoder=skip_encoder,
        lanes=[LaneConfig(seed=201, seats=all_latest_seats(0.0))],
        device="cpu",
        gamma=1.0,
        record_forced_k1_samples=False,
    )

    keep_encoder = _CountingScriptEncoder()
    kept = play_episodes_batched(
        q_nets=_scripted_q_nets(),
        encoder=keep_encoder,
        lanes=[LaneConfig(seed=202, seats=all_latest_seats(0.0))],
        device="cpu",
        gamma=1.0,
        record_forced_k1_samples=True,
    )

    assert skipped == [[]]
    assert skip_encoder.encode_one_calls == 0
    assert len(kept[0]) == 1
    assert kept[0][0].num_legal_actions == 1
    assert keep_encoder.encode_one_calls == 1
    assert [len(env.stepped_actions) for env in envs] == [1, 1]


def test_random_episode_uses_selected_action_encoding_fast_path():
    class CountingEncoder(StateActionEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.encode_all_calls = 0
            self.encode_one_calls = 0

        def encode_all(self, env, player, legal_moves):
            self.encode_all_calls += 1
            return super().encode_all(env, player, legal_moves)

        def encode_one(self, env, player, action, legal_moves):
            self.encode_one_calls += 1
            return super().encode_one(env, player, action, legal_moves)

    encoder = CountingEncoder()
    q_nets = init_guanzero_nets(QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2))
    samples = play_episode(
        q_nets=q_nets,
        encoder=encoder,
        seats=all_latest_seats(1.0),
        seed=13,
        device="cpu",
        gamma=1.0,
    )

    assert samples
    assert encoder.encode_all_calls == 0
    assert encoder.encode_one_calls == len(samples)


# ── Hard-bot mixed-opponent tests ─────────────────────────────────────────


def test_hard_bot_seats_contribute_no_samples():
    """Latest controls {0,2}; hard bot plays seats {1,3}. No seat-1/3 samples."""
    samples = play_episode(
        q_nets=_small_dart_net(),
        encoder=RoleAwareStateActionEncoder(),
        seats=_with_seats(
            all_latest_seats(0.0),
            {1: SeatPolicy.hard_bot(make_agent("strategic")),
             3: SeatPolicy.hard_bot(make_agent("strategic"))},
        ),
        seed=41,
        device="cpu",
        gamma=1.0,
    )
    assert samples, "latest team must produce some training samples"
    assert {s.player for s in samples} <= {0, 2}


def test_hard_bot_self_play_unchanged():
    """hard_bot_seats=frozenset() → all 4 seats in samples (full self-play)."""
    samples = play_episode(
        q_nets=_small_dart_net(),
        encoder=RoleAwareStateActionEncoder(),
        seats=all_latest_seats(1.0),   # random play → terminates fast and exercises every seat
        seed=42,
        device="cpu",
        gamma=1.0,
    )
    assert samples
    assert {s.player for s in samples} == {0, 1, 2, 3}


def test_hard_bot_only_invoked_for_hard_bot_seats():
    """Spy on the bot: every invocation must be for a player in hard_bot_seats.

    Implicit legality check: env.step would raise if the bot returned an
    illegal action, so reaching `assert samples` proves every bot action was
    accepted by the environment.
    """
    real_bot = make_agent("strategic")

    class SpyBot:
        def __init__(self):
            self.players = []

        def act(self, env, player):
            self.players.append(player)
            return real_bot.act(env, player)

    spy = SpyBot()
    samples = play_episode(
        q_nets=_small_dart_net(),
        encoder=RoleAwareStateActionEncoder(),
        seats=_with_seats(
            all_latest_seats(0.0),
            {1: SeatPolicy.hard_bot(spy), 3: SeatPolicy.hard_bot(spy)},
        ),
        seed=43,
        device="cpu",
        gamma=1.0,
    )
    assert samples
    assert spy.players, "hard bot must have been invoked at least once"
    assert set(spy.players) <= {1, 3}


def test_hard_bot_returns_finite_and_attributed_to_latest():
    """Every recorded sample is from a latest-team seat with a finite mc_return.

    Avoids asserting the *sign* of mc_return — we can't guarantee the hard bot
    wins a given game from a single seed. The invariants we *can* guarantee are
    sample attribution and well-formedness of the MC return computation.
    """
    samples = play_episode(
        q_nets=_small_dart_net(),
        encoder=RoleAwareStateActionEncoder(),
        seats=_with_seats(
            all_latest_seats(0.0),
            {1: SeatPolicy.hard_bot(make_agent("strategic")),
             3: SeatPolicy.hard_bot(make_agent("strategic"))},
        ),
        seed=44,
        device="cpu",
        gamma=1.0,
    )
    assert samples
    for s in samples:
        assert s.player in {0, 2}
        assert math.isfinite(s.mc_return)


def test_hard_bot_dispatch_probability():
    """Pure RNG unit test for the worker's single-draw dispatch arithmetic.

    Replicates the worker.py episode-mode draw without touching play_episode
    or the env. With (0.60, 0.40) the observed self-play fraction over 10k
    draws should land in [0.58, 0.62] (well within the binomial 95% band of
    ≈ ±0.0095).
    """
    self_play_frac = 0.60
    hard_bot_frac = 0.40
    rng = random.Random(2026)

    n_self = n_hard = 0
    n = 10_000
    for _ in range(n):
        u = rng.random()
        if u < self_play_frac:
            n_self += 1
        elif u < self_play_frac + hard_bot_frac:
            n_hard += 1
        else:
            n_self += 1

    self_frac = n_self / n
    assert 0.58 <= self_frac <= 0.62, f"self-play frac {self_frac:.3f} outside [0.58, 0.62]"
    assert n_self + n_hard == n


def test_role_random_episode_uses_encode_one_fast_path():
    class CountingEncoder(RoleAwareStateActionEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.encode_all_calls = 0
            self.encode_one_calls = 0

        def encode_all(self, env, player, legal_moves):
            self.encode_all_calls += 1
            return super().encode_all(env, player, legal_moves)

        def encode_one(self, env, player, action, legal_moves):
            self.encode_one_calls += 1
            return super().encode_one(env, player, action, legal_moves)

    encoder = CountingEncoder()
    samples = play_episode(
        q_nets=_small_dart_net(),
        encoder=encoder,
        seats=all_latest_seats(1.0),
        seed=18,
        device="cpu",
        gamma=1.0,
    )

    assert samples
    assert encoder.encode_all_calls == 0
    assert encoder.encode_one_calls == len(samples)


# ── Mixed hard-bot pair tests ─────────────────────────────────────────


def test_pair_dispatch_assigns_distinct_bots():
    """dict-mode hard_bots: each seat gets its own bot instance."""
    real_bot = make_agent("strategic")

    class SpyBot:
        def __init__(self, tag):
            self.tag = tag
            self.players = []

        def act(self, env, player):
            self.players.append(player)
            return real_bot.act(env, player)

    spy_a = SpyBot("A")
    spy_b = SpyBot("B")
    samples = play_episode(
        q_nets=_small_dart_net(),
        encoder=RoleAwareStateActionEncoder(),
        seats=_with_seats(
            all_latest_seats(0.0),
            {1: SeatPolicy.hard_bot(spy_a), 3: SeatPolicy.hard_bot(spy_b)},
        ),
        seed=51,
        device="cpu",
        gamma=1.0,
    )
    assert samples
    # spy_a only saw p=1; spy_b only saw p=3
    assert spy_a.players, "spy_a should have been called"
    assert spy_b.players, "spy_b should have been called"
    assert set(spy_a.players) == {1}
    assert set(spy_b.players) == {3}


def test_pair_episode_samples_only_latest():
    """Mixed-pair vs-hard-bot episode still produces only latest-team samples."""
    samples = play_episode(
        q_nets=_small_dart_net(),
        encoder=RoleAwareStateActionEncoder(),
        seats=_with_seats(
            all_latest_seats(0.0),
            {1: SeatPolicy.hard_bot(make_agent("strategic")),
             3: SeatPolicy.hard_bot(make_agent("yaoji"))},
        ),
        seed=52,
        device="cpu",
        gamma=1.0,
    )
    assert samples
    assert {s.player for s in samples} <= {0, 2}


def test_pair_legacy_agent_still_works():
    """Single-Agent hard_bots (legacy mode) behaves as before — both seats use it."""
    real_bot = make_agent("strategic")

    class SpyBot:
        def __init__(self):
            self.players = []

        def act(self, env, player):
            self.players.append(player)
            return real_bot.act(env, player)

    spy = SpyBot()
    samples = play_episode(
        q_nets=_small_dart_net(),
        encoder=RoleAwareStateActionEncoder(),
        seats=_with_seats(
            all_latest_seats(0.0),
            {1: SeatPolicy.hard_bot(spy), 3: SeatPolicy.hard_bot(spy)},
        ),
        seed=53,
        device="cpu",
        gamma=1.0,
    )
    assert samples
    assert spy.players, "spy bot must have been called"
    assert set(spy.players) <= {1, 3}


def test_pair_sampling_distribution():
    """Pure-RNG unit test for the worker's pair-sampling arithmetic.

    Replicates the worker.py pair-pick draw (after the vs-hard-bot branch is
    selected) without touching play_episode or the env. With 10k pair draws
    each pair's observed fraction must land within ±1.5pp of its weight.
    """
    pair_weights = {
        "jidan_jidan":         0.20,
        "strategic_strategic": 0.10,
        "yaoji_yaoji":         0.25,
        "jidan_strategic":     0.10,
        "jidan_yaoji":         0.25,
        "strategic_yaoji":     0.10,
    }
    names = list(pair_weights.keys())
    weights = list(pair_weights.values())
    rng = random.Random(2027)

    counts = {k: 0 for k in names}
    n = 10_000
    for _ in range(n):
        pick = rng.choices(names, weights=weights, k=1)[0]
        counts[pick] += 1

    for name in names:
        observed = counts[name] / n
        expected = pair_weights[name]
        assert abs(observed - expected) <= 0.015, (
            f"pair {name}: observed {observed:.3f} vs expected {expected:.3f} "
            f"(diff {abs(observed - expected):.3f} > 0.015)"
        )
