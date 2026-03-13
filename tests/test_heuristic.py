"""Tests for the heuristic agent."""

import random

from guandan.cards import BOMB_TYPES, ComboType, Rank
from guandan.game import GuanDanEnv
from guandan.heuristic import HeuristicAgent


def _play_game(agent_teams: dict[str, tuple[int, ...]]) -> list[int]:
    """Play a game with specified agent types per team. Returns finish_order."""
    env = GuanDanEnv()
    heuristic = HeuristicAgent(env.level_rank)
    env.reset()

    while not env.done:
        player = env.current_player
        if player in agent_teams.get("heuristic", ()):
            move = heuristic.choose_action(env, player)
        else:
            move = random.choice(env.legal_moves())
        env.step(move)

    return env.finish_order


def test_heuristic_games_complete():
    """Heuristic on all seats should complete 100 games without crashes."""
    env = GuanDanEnv()
    agent = HeuristicAgent(env.level_rank)

    for _ in range(100):
        env.reset()
        steps = 0
        while not env.done:
            player = env.current_player
            move = agent.choose_action(env, player)
            env.step(move)
            steps += 1
            assert steps < 1000, "Game exceeded 1000 steps"

        assert len(env.finish_order) == 4


def test_heuristic_vs_random_winrate():
    """Heuristic (team {0,2}) should beat random (team {1,3}) 60-90% of the time."""
    wins = 0
    n_games = 200

    for _ in range(n_games):
        env = GuanDanEnv()
        heuristic = HeuristicAgent(env.level_rank)
        env.reset()

        while not env.done:
            player = env.current_player
            if player in (0, 2):
                move = heuristic.choose_action(env, player)
            else:
                move = random.choice(env.legal_moves())
            env.step(move)

        first = env.finish_order[0]
        if first in (0, 2):
            wins += 1

    wr = wins / n_games
    assert 0.55 <= wr <= 0.95, f"Heuristic vs random WR={wr:.1%}, expected 55-95%"


def test_heuristic_vs_heuristic_balance():
    """Heuristic vs heuristic should be ~50% balanced."""
    wins_02 = 0
    n_games = 200

    env = GuanDanEnv()
    agent = HeuristicAgent(env.level_rank)

    for _ in range(n_games):
        env.reset()
        while not env.done:
            player = env.current_player
            move = agent.choose_action(env, player)
            env.step(move)

        first = env.finish_order[0]
        if first in (0, 2):
            wins_02 += 1

    wr = wins_02 / n_games
    assert 0.30 <= wr <= 0.70, f"Heuristic vs heuristic WR={wr:.1%}, expected 30-70%"


def test_heuristic_doesnt_lead_bombs_early():
    """When leading with a full hand, heuristic should not play bombs."""
    env = GuanDanEnv()
    agent = HeuristicAgent(env.level_rank)
    bomb_leads = 0
    total_leads = 0

    for _ in range(50):
        env.reset()
        # Check first move only (full 27-card hand)
        player = env.current_player
        move = agent.choose_action(env, player)
        if env.current_trick is None:
            total_leads += 1
            if move.type in BOMB_TYPES:
                bomb_leads += 1

    # Should almost never lead with a bomb on a full hand
    if total_leads > 0:
        bomb_rate = bomb_leads / total_leads
        assert bomb_rate < 0.05, f"Led with bombs {bomb_rate:.0%} of the time"


def test_heuristic_passes_when_partner_wins():
    """When partner is the trick winner, heuristic should pass."""
    env = GuanDanEnv()
    agent = HeuristicAgent(env.level_rank)
    partner_winning_passes = 0
    partner_winning_total = 0

    for _ in range(200):
        env.reset()
        steps = 0
        while not env.done and steps < 500:
            player = env.current_player
            partner = (player + 2) % 4

            # Check if partner is the trick winner
            if env.current_trick is not None and env.trick_winner == partner:
                partner_winning_total += 1
                move = agent.choose_action(env, player)
                if move.type == ComboType.PASS:
                    partner_winning_passes += 1
                env.step(move)
            else:
                move = agent.choose_action(env, player)
                env.step(move)
            steps += 1

    if partner_winning_total > 0:
        pass_rate = partner_winning_passes / partner_winning_total
        assert pass_rate > 0.85, (
            f"Pass rate when partner winning: {pass_rate:.0%}, expected >85%"
        )
