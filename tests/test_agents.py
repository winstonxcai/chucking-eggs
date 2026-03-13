"""Tests for the agent hierarchy."""

import random

from guandan.agents import (
    GreedyBot,
    HeuristicBot,
    RandomBot,
    StrategicBot,
    make_agent,
)
from guandan.cards import BOMB_TYPES, ComboType, Rank
from guandan.game import GuanDanEnv


def _play_tournament(agent_a, agent_b, n_games: int = 500) -> float:
    """Play n_games with agent_a on team {0,2} vs agent_b on team {1,3}.

    Returns win rate for team {0,2}.
    """
    env = GuanDanEnv()
    wins = 0

    for _ in range(n_games):
        env.reset()
        steps = 0
        while not env.done:
            player = env.current_player
            if player in (0, 2):
                move = agent_a.act(env, player)
            else:
                move = agent_b.act(env, player)
            env.step(move)
            steps += 1
            assert steps < 1000, "Game exceeded 1000 steps"

        first = env.finish_order[0]
        if first in (0, 2):
            wins += 1

    return wins / n_games


def test_all_agents_complete_games():
    """Each agent type should complete 500 games on all seats without crashes."""
    env = GuanDanEnv()
    agents = [
        RandomBot(),
        GreedyBot(env.level_rank),
        HeuristicBot(env.level_rank),
        StrategicBot(env.level_rank),
    ]

    for agent in agents:
        for _ in range(500):
            env.reset()
            steps = 0
            while not env.done:
                player = env.current_player
                move = agent.act(env, player)
                env.step(move)
                steps += 1
                assert steps < 1000, (
                    f"{agent.__class__.__name__} game exceeded 1000 steps"
                )
            assert len(env.finish_order) == 4


def test_make_agent_factory():
    """make_agent should create all registered agent types."""
    for name in ("random", "greedy", "heuristic", "strategic"):
        agent = make_agent(name)
        assert agent is not None


def test_greedy_beats_random():
    """GreedyBot should beat RandomBot with >55% win rate."""
    greedy = GreedyBot(Rank.TWO)
    rand = RandomBot()
    wr = _play_tournament(greedy, rand, n_games=500)
    assert wr > 0.55, f"Greedy vs Random WR={wr:.1%}, expected >55%"


def test_heuristic_beats_greedy():
    """HeuristicBot should beat GreedyBot with >52% win rate."""
    heur = HeuristicBot(Rank.TWO)
    greedy = GreedyBot(Rank.TWO)
    wr = _play_tournament(heur, greedy, n_games=500)
    assert wr > 0.52, f"Heuristic vs Greedy WR={wr:.1%}, expected >52%"


def test_strategic_beats_heuristic():
    """StrategicBot should beat HeuristicBot with >52% win rate."""
    strat = StrategicBot(Rank.TWO)
    heur = HeuristicBot(Rank.TWO)
    wr = _play_tournament(strat, heur, n_games=500)
    assert wr > 0.52, f"Strategic vs Heuristic WR={wr:.1%}, expected >52%"


def test_greedy_never_bombs_following():
    """GreedyBot should never play bombs when following."""
    env = GuanDanEnv()
    greedy = GreedyBot(env.level_rank)
    bomb_follows = 0

    for _ in range(500):
        env.reset()
        steps = 0
        while not env.done and steps < 500:
            player = env.current_player
            move = greedy.act(env, player)
            if env.current_trick is not None and move.type in BOMB_TYPES:
                bomb_follows += 1
            env.step(move)
            steps += 1

    assert bomb_follows == 0, (
        f"GreedyBot bombed {bomb_follows} times while following"
    )


def test_strategic_passes_when_partner_wins():
    """StrategicBot should pass when partner is winning the trick."""
    env = GuanDanEnv()
    agent = StrategicBot(env.level_rank)
    partner_winning_passes = 0
    partner_winning_total = 0

    for _ in range(500):
        env.reset()
        steps = 0
        while not env.done and steps < 500:
            player = env.current_player
            partner = (player + 2) % 4

            if env.current_trick is not None and env.trick_winner == partner:
                partner_winning_total += 1
                move = agent.act(env, player)
                if move.type == ComboType.PASS:
                    partner_winning_passes += 1
                env.step(move)
            else:
                move = agent.act(env, player)
                env.step(move)
            steps += 1

    if partner_winning_total > 0:
        pass_rate = partner_winning_passes / partner_winning_total
        assert pass_rate > 0.85, (
            f"Pass rate when partner winning: {pass_rate:.0%}, expected >85%"
        )
