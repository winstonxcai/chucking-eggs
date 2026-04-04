"""Round-robin win rate matrix across all bot agents.

Runs every bot pair (A on seats {0,2} vs B on seats {1,3}) and records
win rates. Then derives calibrated Glicko-2 ratings from the empirical WRs.

Usage:
    PYTHONPATH=src python scripts/wr_matrix.py --games 200
    PYTHONPATH=src python scripts/wr_matrix.py --games 500 --agents random,greedy,heuristic,strategic
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from guandan.agents import RLAgentLSTM, make_agent
from guandan.cards import Rank
from guandan.game import GuanDanEnv
from guandan.rating import GlickoPlayer, glicko2_update
from guandan.training.q_network import QNetworkLSTM, get_device

DEFAULT_AGENTS = ["random", "greedy", "xingdream", "heuristic", "strategic", "lalala", "noai"]
# RL excluded from default list — add with --agents and --rl-checkpoint.


def run_matchup(agent_a, agent_b, n_games: int) -> dict:
    """Run n_games with agent_a on seats {0,2} vs agent_b on seats {1,3}.

    Returns dict with wins_a, wins_b, draws, winrate_a.
    """
    env = GuanDanEnv()
    wins_a = 0
    total_r = 0.0

    for _ in range(n_games):
        env.reset()
        while not env.done:
            p = env.current_player
            if p in (0, 2):
                move = agent_a.act(env, p)
            else:
                move = agent_b.act(env, p)
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
    """Derive Glicko-2 ratings from the empirical WR matrix.

    Runs multiple passes over the matrix, treating each pass as a rating period
    where each bot plays all its matchups.
    """
    players = {name: GlickoPlayer(name=name) for name in agent_names}

    for pass_num in range(n_passes):
        new_players = {}
        for name in agent_names:
            opponents = []
            outcomes = []
            for opp_name in agent_names:
                if opp_name == name:
                    continue
                data = matrix[name][opp_name]
                n = data["n_games"]
                w = data["wins"]
                # Expand into individual game outcomes for Glicko-2
                # Use aggregate: treat as n encounters with outcome w/n each
                # More efficient: single update with fractional score
                opponents.append(players[opp_name])
                outcomes.append(w / n)

            new_players[name] = glicko2_update(players[name], opponents, outcomes)
            new_players[name].name = name

        players = new_players

    return players


def print_matrix(matrix: dict[str, dict[str, dict]], agent_names: list[str]) -> None:
    """Print formatted WR matrix table."""
    # Header
    max_name = max(len(n) for n in agent_names)
    col_w = 7
    header = " " * (max_name + 1) + "".join(f"{n:>{col_w}}" for n in agent_names)
    print()
    print("Win Rate Matrix (row = team {0,2}, col = team {1,3})")
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
    """Print derived ratings sorted by strength."""
    sorted_bots = sorted(agent_names, key=lambda n: ratings[n].rating, reverse=True)
    print("Derived Glicko-2 Ratings")
    print("=" * 40)
    print(f"{'Bot':>12}  {'Rating':>7}  {'RD':>5}")
    print("-" * 40)
    for name in sorted_bots:
        p = ratings[name]
        print(f"{name:>12}  {p.rating:>7.0f}  {p.rd:>5.0f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-robin WR matrix for all bots")
    parser.add_argument("--games", type=int, default=200,
                        help="Games per directed matchup (default: 200)")
    parser.add_argument("--agents", type=str, default=None,
                        help="Comma-separated agent names (default: all rule-based)")
    parser.add_argument("--rl-checkpoint", type=str, default=None,
                        help="Path to RL checkpoint .pt file (required if 'rl' in agents)")
    parser.add_argument("--output", type=str, default="runs/wr_matrix",
                        help="Output directory (default: runs/wr_matrix)")
    parser.add_argument("--rating-passes", type=int, default=30,
                        help="Glicko-2 convergence passes (default: 30)")
    args = parser.parse_args()

    agent_names = args.agents.split(",") if args.agents else DEFAULT_AGENTS
    n_games = args.games
    level_rank = Rank.TWO

    # Total matchups estimate
    n_matchups = len(agent_names) * (len(agent_names) - 1)
    total_games = n_matchups * n_games
    print(f"Round-robin: {len(agent_names)} agents, {n_matchups} matchups, "
          f"{total_games} total games @ {n_games}/matchup")

    # Create agents
    agents = {}
    for name in agent_names:
        if name == "rl":
            if not args.rl_checkpoint:
                print("ERROR: 'rl' in agents but --rl-checkpoint not provided")
                sys.exit(1)
            ckpt_path = Path(args.rl_checkpoint)
            if not ckpt_path.exists():
                print(f"ERROR: checkpoint not found: {ckpt_path}")
                sys.exit(1)
            device = get_device()
            q_lead = QNetworkLSTM(lstm_hidden=256, hidden=1024, use_gnn=True, gnn_out=128).to(device)
            q_follow = QNetworkLSTM(lstm_hidden=256, hidden=1024, use_gnn=True, gnn_out=128).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            lead_key = "lead_state_dict" if "lead_state_dict" in ckpt else "lead"
            follow_key = "follow_state_dict" if "follow_state_dict" in ckpt else "follow"
            q_lead.load_state_dict(ckpt[lead_key], strict=False)
            q_follow.load_state_dict(ckpt[follow_key], strict=False)
            q_lead.eval()
            q_follow.eval()
            ep = ckpt.get("episode", "?")
            agents["rl"] = RLAgentLSTM(q_lead, q_follow, device)
            print(f"  RL: loaded {ckpt_path.name} (ep {ep})")
        else:
            agents[name] = make_agent(name, level_rank=level_rank)
    print(f"Agents loaded: {', '.join(agent_names)}")

    # Run all matchups
    matrix: dict[str, dict[str, dict]] = {a: {} for a in agent_names}
    t_start = time.time()
    completed = 0

    for i, a in enumerate(agent_names):
        for j, b in enumerate(agent_names):
            if a == b:
                continue
            t0 = time.time()
            result = run_matchup(agents[a], agents[b], n_games)
            elapsed = time.time() - t0
            matrix[a][b] = result
            completed += 1
            print(f"  [{completed}/{n_matchups}] {a} vs {b}: "
                  f"{result['winrate']:.1%}  ({elapsed:.1f}s)")

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f}min)")

    # Print matrix
    print_matrix(matrix, agent_names)

    # Derive ratings
    ratings = derive_ratings(matrix, agent_names, n_passes=args.rating_passes)
    print_ratings(ratings, agent_names)

    # Save results
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Serialize matrix (convert for JSON)
    matrix_json = {}
    for a in agent_names:
        matrix_json[a] = {}
        for b in agent_names:
            if a != b:
                matrix_json[a][b] = matrix[a][b]

    results = {
        "agents": agent_names,
        "games_per_matchup": n_games,
        "total_games": total_games,
        "total_time_s": round(total_time, 1),
        "matrix": matrix_json,
        "ratings": {name: ratings[name].to_dict() for name in agent_names},
    }

    out_file = out_dir / "results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
