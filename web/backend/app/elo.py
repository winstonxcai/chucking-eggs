"""Team-aware Elo for 2v2 Guan Dan.

Teams: seats 0+2 (team A) vs seats 1+3 (team B).
Bot Elo anchors are calibrated Glicko-2 ratings from a 13-bot WR matrix.
"""

from __future__ import annotations

# Bot Elo anchors by difficulty (calibrated, fixed — bots never update).
# Values sourced from BOT_POOLS in ai_service.py.
BOT_ELOS: dict[str, int] = {
    "wjsd": 1212,
    "liuzha": 1260,
    "hulalala": 1264,
    "easy": 1415,
    "competition": 1464,
    "casual": 1523,
    "hard": 1621,
    "master": 1726,
    "yaoji": 1772,
    "jidan": 1779,
    "expert": 1786,
}

# Bot entries injected server-side into the leaderboard.
BOT_LEADERBOARD_ENTRIES: list[dict] = [
    {"username": "Expert", "elo": 1786, "games_played": None, "is_bot": True},
    {"username": "Jidan", "elo": 1779, "games_played": None, "is_bot": True},
    {"username": "Yaoji", "elo": 1772, "games_played": None, "is_bot": True},
    {"username": "NoAI", "elo": 1726, "games_played": None, "is_bot": True},
    {"username": "Tiger / Falcon / Leopard", "elo": 1621, "games_played": None, "is_bot": True},
    {"username": "Panda / Owl / Cat (Medium)", "elo": 1523, "games_played": None, "is_bot": True},
    {"username": "Lalala", "elo": 1464, "games_played": None, "is_bot": True},
    {"username": "Koala / Turtle / Lamb", "elo": 1415, "games_played": None, "is_bot": True},
    {"username": "Hulalala", "elo": 1264, "games_played": None, "is_bot": True},
    {"username": "Liuzha", "elo": 1260, "games_played": None, "is_bot": True},
    {"username": "Wjsd", "elo": 1212, "games_played": None, "is_bot": True},
]


def _k_factor(games_played: int) -> float:
    if games_played < 30:
        return 40.0
    if games_played < 100:
        return 24.0
    return 16.0


# Per-position actual values: 1st=1.0, 2nd=0.75, 3rd=0.25, 4th=0.0.
# Sum is 2.0, matching the sum of expected probabilities across all 4 players,
# so the system remains zero-sum at even matchups.
_POSITION_ACTUAL = [1.0, 0.75, 0.25, 0.0]


def _margin_multiplier(reward: float) -> float:
    """Map reward from get_rewards() to a K multiplier.

    Rewards: ±3 = 双上 (both teammates finish 1st+2nd),
             ±1 = normal win/loss.
    """
    if abs(reward) >= 3:
        return 1.5
    return 1.0


def _expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def compute_elo_delta(
    player_elo: int,
    partner_elo: int,
    opp1_elo: int,
    opp2_elo: int,
    player_games: int,
    finish_pos: int,
    reward: float,
) -> int:
    """Return the integer Elo delta for one player after a completed game.

    finish_pos: 0-indexed finish position (0=1st, 1=2nd, 2=3rd, 3=4th).
    """
    team_rating = (player_elo + partner_elo) / 2.0
    opp_rating = (opp1_elo + opp2_elo) / 2.0
    expected = _expected(team_rating, opp_rating)
    actual = _POSITION_ACTUAL[finish_pos]
    k = _k_factor(player_games) * _margin_multiplier(reward)
    return round(k * (actual - expected))
