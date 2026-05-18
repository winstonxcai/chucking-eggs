from __future__ import annotations

import random

from guandan.dart.config import EpisodeMixConfig
from guandan.dart.data.sample_tags import (
    EPISODE_MODE_SELF_PLAY,
    EPISODE_MODE_VS_CHECKPOINT,
    EPISODE_MODE_VS_HARD_BOT,
    OPPONENT_CHECKPOINT_BASE,
    OPPONENT_STRATEGIC,
    OPPONENT_YAOJI,
)
from guandan.dart.runtime.actor.opponents import OpponentPools


class DummyBot:
    pass


def _pools(
    *,
    frozen_nets=(),
    hard_bots_pool=(),
    hard_bot_weights=(),
    pair_names=(),
    pair_weights=(),
    self_play=0.0,
    frozen_pool=0.0,
    hard_bot=1.0,
    latest_team_odd_probability=0.0,
) -> OpponentPools:
    return OpponentPools(
        frozen_nets=tuple(frozen_nets),
        frozen_checkpoint_paths=tuple(f"ckpt_{i}.pt" for i, _ in enumerate(frozen_nets)),
        frozen_epsilon=0.0,
        hard_bots_pool=tuple(hard_bots_pool),
        hard_bots_by_name={name: bot for name, bot in hard_bots_pool},
        hard_bot_weights=tuple(hard_bot_weights),
        pair_names=tuple(pair_names),
        pair_weights=tuple(pair_weights),
        episode_mix=EpisodeMixConfig(
            self_play=self_play,
            frozen_pool=frozen_pool,
            hard_bot=hard_bot,
        ),
        latest_team_odd_probability=latest_team_odd_probability,
        mode_counts={"self_play": 0, "vs_frozen": 0, "vs_hard_bot": 0},
        team_counts={"latest_even": 0, "latest_odd": 0},
        frozen_pick_counts=[0] * len(frozen_nets),
        hard_bot_pick_counts=[0] * len(hard_bots_pool),
        pair_pick_counts={key: 0 for key in pair_names},
    )


def test_empty_opponent_pools_sample_self_play():
    pools = _pools(self_play=1.0, hard_bot=0.0)

    seats, tags = pools.sample_lane(random.Random(1), eps=0.25)

    assert tags.mode == EPISODE_MODE_SELF_PLAY
    assert [seat.kind for seat in seats] == ["latest"] * 4
    assert [seat.epsilon for seat in seats] == [0.25] * 4
    assert pools.mode_counts == {"self_play": 1, "vs_frozen": 0, "vs_hard_bot": 0}


def test_frozen_pool_assigns_checkpoint_to_opposing_team():
    frozen = object()
    pools = _pools(
        frozen_nets=(frozen,),
        frozen_pool=1.0,
        hard_bot=0.0,
        latest_team_odd_probability=1.0,
    )

    seats, tags = pools.sample_lane(random.Random(2), eps=0.1)

    assert tags.mode == EPISODE_MODE_VS_CHECKPOINT
    assert tags.opponent_id == OPPONENT_CHECKPOINT_BASE
    assert tags.latest_team == 1
    assert [seat.kind for seat in seats] == ["frozen", "latest", "frozen", "latest"]
    assert seats[0].frozen_net is frozen
    assert seats[2].frozen_net is frozen
    assert pools.mode_counts["vs_frozen"] == 1
    assert pools.team_counts["latest_odd"] == 1
    assert pools.frozen_pick_counts == [1]


def test_frozen_pool_counts_multiple_checkpoints():
    pools = _pools(
        frozen_nets=(object(), object()),
        frozen_pool=1.0,
        hard_bot=0.0,
        latest_team_odd_probability=0.0,
    )
    rng = random.Random(3)

    for _ in range(20):
        pools.sample_lane(rng, eps=0.1)

    assert sum(pools.frozen_pick_counts) == 20
    assert pools.mode_counts["vs_frozen"] == 20
    assert pools.team_counts["latest_even"] == 20


def test_hard_bot_uniform_sampling_uses_empty_weights_as_uniform():
    bot_a = DummyBot()
    bot_b = DummyBot()
    pools = _pools(
        hard_bots_pool=(("strategic", bot_a), ("yaoji", bot_b)),
        hard_bot_weights=(),
        latest_team_odd_probability=0.0,
    )

    seats, tags = pools.sample_lane(random.Random(4), eps=0.1)

    assert tags.mode == EPISODE_MODE_VS_HARD_BOT
    assert tags.latest_team == 0
    assert [seat.kind for seat in seats] == ["latest", "hard_bot", "latest", "hard_bot"]
    assert sum(pools.hard_bot_pick_counts) == 1
    assert pools.mode_counts["vs_hard_bot"] == 1


def test_hard_bot_weighted_sampling_counts_selected_bot():
    bot_a = DummyBot()
    bot_b = DummyBot()
    pools = _pools(
        hard_bots_pool=(("strategic", bot_a), ("yaoji", bot_b)),
        hard_bot_weights=(1.0, 0.0),
    )
    rng = random.Random(5)

    for _ in range(10):
        seats, tags = pools.sample_lane(rng, eps=0.1)
        assert tags.opponent_id == OPPONENT_STRATEGIC
        assert seats[1].hard_bot_agent is bot_a
        assert seats[3].hard_bot_agent is bot_a

    assert pools.hard_bot_pick_counts == [10, 0]


def test_hard_bot_pair_sampling_sets_pair_counter_and_yaoji_target():
    bot_a = DummyBot()
    bot_b = DummyBot()
    pools = _pools(
        hard_bots_pool=(("strategic", bot_a), ("yaoji", bot_b)),
        pair_names=("strategic_yaoji",),
        pair_weights=(1.0,),
        latest_team_odd_probability=1.0,
    )

    seats, tags = pools.sample_lane(random.Random(6), eps=0.1)

    assert tags.mode == EPISODE_MODE_VS_HARD_BOT
    assert tags.opponent_id == OPPONENT_YAOJI
    assert tags.latest_team == 1
    assert seats[0].kind == "hard_bot"
    assert seats[2].kind == "hard_bot"
    assert pools.pair_pick_counts == {"strategic_yaoji": 1}


def test_hard_bot_pool_can_still_sample_self_play():
    pools = _pools(
        hard_bots_pool=(("strategic", DummyBot()),),
        self_play=1.0,
        hard_bot=0.0,
    )

    seats, tags = pools.sample_lane(random.Random(7), eps=0.1)

    assert tags.mode == EPISODE_MODE_SELF_PLAY
    assert [seat.kind for seat in seats] == ["latest"] * 4
    assert pools.mode_counts == {"self_play": 1, "vs_frozen": 0, "vs_hard_bot": 0}
    assert pools.hard_bot_pick_counts == [0]


def test_frozen_and_hard_bot_pools_can_both_be_active():
    pools = _pools(
        frozen_nets=(object(),),
        hard_bots_pool=(("strategic", DummyBot()),),
        self_play=0.0,
        frozen_pool=0.5,
        hard_bot=0.5,
    )
    rng = random.Random(8)

    for _ in range(80):
        pools.sample_lane(rng, eps=0.1)

    assert pools.mode_counts["vs_frozen"] > 0
    assert pools.mode_counts["vs_hard_bot"] > 0
    assert pools.mode_counts["self_play"] == 0


def test_selected_empty_pool_falls_back_to_self_play():
    pools = _pools(self_play=0.0, frozen_pool=1.0, hard_bot=0.0)

    seats, tags = pools.sample_lane(random.Random(9), eps=0.1)

    assert tags.mode == EPISODE_MODE_SELF_PLAY
    assert [seat.kind for seat in seats] == ["latest"] * 4
    assert pools.mode_counts == {"self_play": 1, "vs_frozen": 0, "vs_hard_bot": 0}
