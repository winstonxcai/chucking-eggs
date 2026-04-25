"""Round-robin win rate matrix across all bot agents.

Runs every ordered bot pair (A on seats {0,2} vs B on seats {1,3}) and
records win rates. Derives calibrated Glicko-2 ratings from the empirical WRs.

Use --inject to fold in pre-known matchup results without re-running them:
    --inject "partner_oracle,jidan,150,200"   (name, opponent, wins, n_games)
  Injected agents are included in Glicko derivation using only their known matchups.

Usage:
    PYTHONPATH=ml/src python ml/scripts/eval/wr_matrix.py --games 200
    PYTHONPATH=ml/src python ml/scripts/eval/wr_matrix.py --games 200 --agents greedy,heuristic,strategic,jidan
    PYTHONPATH=ml/src python ml/scripts/eval/wr_matrix.py --games 200 \\
        --inject "partner_oracle,jidan,150,200"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from guandan.agents import PartnerOracleBot, PartnerPIMCBot, make_agent
from guandan.cards import Rank
from guandan.game import GuanDanEnv
from guandan.rating import GlickoPlayer, glicko2_update

DEFAULT_AGENTS = [
    "random", "greedy", "heuristic", "strategic",
    "xingdream", "yaoji", "jidan",
]


def build_agent(
    name: str,
    level_rank: int,
    n_det: int,
    n_cands: int,
    checkpoint: str | None = None,
    top_k: int = 3,
):
    if name == "partner_pimc":
        return PartnerPIMCBot(level_rank=level_rank, n_det=n_det, n_cands=n_cands)
    if name == "partner_oracle":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for partner_oracle")
        return PartnerOracleBot(
            checkpoint_path=checkpoint,
            level_rank=level_rank,
            use_search=True,
            n_det=n_det,
            top_k=top_k,
            use_belief=True,
        )
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
        "injected": False,
    }


def parse_inject(inject_strs: list[str]) -> dict[str, dict[str, dict]]:
    """Parse --inject entries into matrix-compatible dicts.

    Each entry: "name,opponent,wins,n_games"
    Automatically adds the symmetric reverse entry.
    """
    injected: dict[str, dict[str, dict]] = {}
    for s in inject_strs:
        parts = s.strip().split(",")
        if len(parts) != 4:
            raise ValueError(f"--inject entry must be 'name,opponent,wins,n_games': {s!r}")
        name, opp, wins_s, n_s = parts
        wins, n = int(wins_s), int(n_s)
        losses = n - wins

        for a, b, w, l in [(name, opp, wins, losses), (opp, name, losses, wins)]:
            injected.setdefault(a, {})[b] = {
                "wins": w, "losses": l, "n_games": n,
                "winrate": w / n, "avg_reward": 0.0, "injected": True,
            }
    return injected


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
                data = matrix.get(name, {}).get(opp_name)
                if data is None:
                    continue  # skip missing pairs (injected agents with partial data)
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
            elif b not in matrix.get(a, {}):
                row += f"{'???':>{col_w}}"
            else:
                wr = matrix[a][b]["winrate"]
                inj = "*" if matrix[a][b].get("injected") else ""
                row += f"{wr:>{col_w-1}.1%}{inj}"
        print(row)
    print("  (* = injected from prior eval, not re-run)")
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
                        help="partner_pimc/partner_oracle: determinizations per move (default 20)")
    parser.add_argument("--n-cands", type=int, default=10,
                        help="partner_pimc: max candidates pre-filter (default 10)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="partner_oracle: top-K candidates from policy (default 3)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for partner_oracle agent")
    parser.add_argument("--inject", type=str, action="append", default=[],
                        help="Pre-known result: 'name,opponent,wins,n_games'. "
                             "Agent is added to Glicko without re-running games. "
                             "Reverse is auto-inferred. Can be repeated.")
    parser.add_argument("--output", type=str, default="ml/runs/wr_matrix",
                        help="Output directory (default: ml/runs/wr_matrix)")
    parser.add_argument("--rating-passes", type=int, default=30)
    args = parser.parse_args()

    agent_names = args.agents.split(",") if args.agents else DEFAULT_AGENTS[:]

    # Parse injected results. Only agents NOT already in agent_names are injected-only.
    injected_matrix = parse_inject(args.inject)
    injected_only = set()  # agents that exist only via --inject (no live games)
    for inj_name in injected_matrix:
        if inj_name not in agent_names:
            agent_names.append(inj_name)
            injected_only.add(inj_name)

    n_games = args.games
    level_rank = Rank.TWO

    live_names = [n for n in agent_names if n not in injected_only]

    n_matchups = len(live_names) * (len(live_names) - 1)
    total_games = n_matchups * n_games
    print(f"Round-robin: {len(live_names)} live agents, {n_matchups} matchups, "
          f"{total_games} total games @ {n_games}/matchup")
    if injected_only:
        print(f"Injected (no games run): {', '.join(sorted(injected_only))}")

    agents = {
        name: build_agent(name, level_rank, args.n_det, args.n_cands,
                          checkpoint=args.checkpoint, top_k=args.top_k)
        for name in live_names
    }
    print(f"Agents loaded: {', '.join(live_names)}")

    matrix: dict[str, dict[str, dict]] = {a: {} for a in agent_names}
    # Pre-populate injected results
    for a, row in injected_matrix.items():
        matrix.setdefault(a, {}).update(row)

    t_start = time.time()
    completed = 0

    for a in live_names:
        for b in live_names:
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
        "matrix": {a: {b: matrix[a][b] for b in agent_names if b != a and b in matrix.get(a, {})}
                   for a in agent_names},
        "ratings": {name: ratings[name].to_dict() for name in agent_names},
    }
    out_file = out_dir / f"results_{time.strftime('%m%d_%H%M')}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
