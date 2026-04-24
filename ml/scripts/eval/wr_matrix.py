"""Round-robin win rate matrix across all bot agents.

Runs every ordered bot pair (A on seats {0,2} vs B on seats {1,3}) and
records win rates. Derives calibrated Glicko-2 ratings from the empirical WRs.

Usage:
    PYTHONPATH=ml/src python ml/scripts/eval/wr_matrix.py --games 200
    PYTHONPATH=ml/src python ml/scripts/eval/wr_matrix.py --games 200 --agents greedy,heuristic,strategic,jidan
    PYTHONPATH=ml/src python ml/scripts/eval/wr_matrix.py --games 50 --agents strategic,jidan,partner_pimc --n-det 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from guandan.agents import PartnerPIMCBot, make_agent
from guandan.cards import Rank
from guandan.game import GuanDanEnv
from guandan.rating import GlickoPlayer, glicko2_update

DEFAULT_AGENTS = [
    "random", "greedy", "heuristic", "strategic",
    "xingdream", "lalala", "noai", "yaoji", "jidan",
]


def build_agent(name: str, level_rank: int, n_det: int, n_cands: int):
    if name == "partner_pimc":
        return PartnerPIMCBot(level_rank=level_rank, n_det=n_det, n_cands=n_cands)
    return make_agent(name, level_rank=level_rank)


def run_matchup(agent_a, agent_b, n_games: int) -> dict:
    """Run n_games with agent_a on seats {0,2} vs agent_b on seats {1,3}."""
    env = GuanDanEnv()
    wins_a = 0
    total_r = 0.0

    for _ in range(n_games):
        env.reset()
        while not env.done:
            p = env.current_player
            move = agent_a.act(env, p) if p in (0, 2) else agent_b.act(env, p)
            env.step(move)

        rewards = env.get_rewards()
        team_r = rewards[0] + rewards[2]
        if team_r > 0:
            wins_a += 1
        total_r += team_r

    return {
        "wins": wins_a,
        "losses": n_games - wins_a,
        "n_games": n_games,
        "winrate": wins_a / n_games,
        "avg_reward": total_r / n_games,
    }


def derive_ratings(
    matrix: dict[str, dict[str, dict]],
    agent_names: list[str],
    n_passes: int = 30,
) -> dict[str, GlickoPlayer]:
    players = {name: GlickoPlayer(name=name) for name in agent_names}

    for _ in range(n_passes):
        new_players = {}
        for name in agent_names:
            opponents = []
            outcomes = []
            for opp_name in agent_names:
                if opp_name == name:
                    continue
                data = matrix[name][opp_name]
                opponents.append(players[opp_name])
                outcomes.append(data["wins"] / data["n_games"])
            new_players[name] = glicko2_update(players[name], opponents, outcomes)
            new_players[name].name = name
        players = new_players

    return players


def print_matrix(matrix: dict[str, dict[str, dict]], agent_names: list[str]) -> None:
    max_name = max(len(n) for n in agent_names)
    col_w = 8
    header = " " * (max_name + 1) + "".join(f"{n:>{col_w}}" for n in agent_names)
    print()
    print("Win Rate Matrix (row = seats {0,2}, col = seats {1,3})")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for a in agent_names:
        row = f"{a:>{max_name}} "
        for b in agent_names:
            if a == b:
                row += f"{'---':>{col_w}}"
            else:
                wr = matrix[a][b]["winrate"]
                row += f"{wr:>{col_w}.1%}"
        print(row)
    print()


def print_ratings(ratings: dict[str, GlickoPlayer], agent_names: list[str]) -> None:
    sorted_bots = sorted(agent_names, key=lambda n: ratings[n].rating, reverse=True)
    print("Derived Glicko-2 Ratings")
    print("=" * 40)
    print(f"{'Bot':>14}  {'Rating':>7}  {'RD':>5}")
    print("-" * 40)
    for name in sorted_bots:
        p = ratings[name]
        print(f"{name:>14}  {p.rating:>7.0f}  {p.rd:>5.0f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-robin WR matrix for all bots")
    parser.add_argument("--games", type=int, default=200,
                        help="Games per directed matchup (default: 200)")
    parser.add_argument("--agents", type=str, default=None,
                        help="Comma-separated agent names (default: all rule-based)")
    parser.add_argument("--n-det", type=int, default=20,
                        help="partner_pimc: determinizations per move (default 20)")
    parser.add_argument("--n-cands", type=int, default=10,
                        help="partner_pimc: max candidates pre-filter (default 10)")
    parser.add_argument("--output", type=str, default="ml/runs/wr_matrix",
                        help="Output directory (default: ml/runs/wr_matrix)")
    parser.add_argument("--rating-passes", type=int, default=30)
    args = parser.parse_args()

    agent_names = args.agents.split(",") if args.agents else DEFAULT_AGENTS
    n_games = args.games
    level_rank = Rank.TWO

    n_matchups = len(agent_names) * (len(agent_names) - 1)
    total_games = n_matchups * n_games
    print(f"Round-robin: {len(agent_names)} agents, {n_matchups} matchups, "
          f"{total_games} total games @ {n_games}/matchup")

    agents = {
        name: build_agent(name, level_rank, args.n_det, args.n_cands)
        for name in agent_names
    }
    print(f"Agents loaded: {', '.join(agent_names)}")

    matrix: dict[str, dict[str, dict]] = {a: {} for a in agent_names}
    t_start = time.time()
    completed = 0

    for a in agent_names:
        for b in agent_names:
            if a == b:
                continue
            t0 = time.time()
            result = run_matchup(agents[a], agents[b], n_games)
            elapsed = time.time() - t0
            matrix[a][b] = result
            completed += 1
            print(f"  [{completed:3d}/{n_matchups}] {a:>14} vs {b:<14}  "
                  f"WR={result['winrate']:.1%}  ({elapsed:.1f}s)")

    total_time = time.time() - t_start
    print(f"\nTotal: {total_time:.0f}s ({total_time/60:.1f}min)")

    print_matrix(matrix, agent_names)

    ratings = derive_ratings(matrix, agent_names, n_passes=args.rating_passes)
    print_ratings(ratings, agent_names)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "agents": agent_names,
        "games_per_matchup": n_games,
        "total_games": total_games,
        "total_time_s": round(total_time, 1),
        "matrix": {a: {b: matrix[a][b] for b in agent_names if b != a} for a in agent_names},
        "ratings": {name: ratings[name].to_dict() for name in agent_names},
    }
    out_file = out_dir / f"results_{time.strftime('%m%d_%H%M')}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
