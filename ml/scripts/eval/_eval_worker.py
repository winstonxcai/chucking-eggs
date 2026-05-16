"""Shared game-playing utilities for eval scripts.

Provides ``play_n_games`` — a single function that runs N games with agent A
on seats {0, 2} and agent B on seats {1, 3} and returns win/loss counts. Both
``eval_dart.py`` and ``wr_matrix.py`` import from here so the game-loop
logic has one canonical implementation.
"""

from __future__ import annotations

from typing import Any


def play_n_games(
    agent_a: Any,
    agent_b: Any,
    n_games: int,
    seeds: list[int] | None = None,
    label: str = "",
    progress: int = 0,
) -> dict[str, Any]:
    """Play ``n_games`` with agent_a on seats {0, 2} and agent_b on {1, 3}.

    If ``seeds`` is provided it must have ``n_games`` elements; each game is
    reset with the corresponding seed so decks are reproducible.

    Returns a dict with keys: ``wins``, ``losses``, ``n_games``, ``winrate``,
    ``avg_reward``.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

    from guandan.game import GuanDanEnv

    env = GuanDanEnv()
    wins_a = 0
    total_r = 0.0

    for i in range(n_games):
        kw = {"seed": seeds[i]} if seeds is not None else {}
        env.reset(**kw)
        while not env.done:
            p = env.current_player
            move = agent_a.act(env, p) if p in (0, 2) else agent_b.act(env, p)
            env.step(move)
        rewards = env.get_rewards()
        team_r = rewards[0] + rewards[2]
        if team_r > 0:
            wins_a += 1
        total_r += team_r
        if progress and label and (i + 1) % progress == 0:
            print(f"  {label}: {i+1}/{n_games}  WR={wins_a/(i+1):.1%}", flush=True)

    return {
        "wins": wins_a,
        "losses": n_games - wins_a,
        "n_games": n_games,
        "winrate": wins_a / n_games,
        "avg_reward": total_r / n_games,
    }
