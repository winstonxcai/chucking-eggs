"""Round-robin win rate matrix across all bot agents.

Runs every ordered bot pair (A on seats {0,2} vs B on seats {1,3}) and
records win rates. Derives calibrated Glicko-2 ratings from the empirical WRs.

Use --inject to fold in pre-known matchup results without re-running them:
    --inject "name,opponent,wins,n_games"
  Injected agents are included in Glicko derivation using only their known matchups.

Usage:
    PYTHONPATH=ml/src python ml/scripts/eval/wr_matrix.py --games 200
    PYTHONPATH=ml/src python ml/scripts/eval/wr_matrix.py --games 200 --agents greedy,heuristic,strategic,jidan
    PYTHONPATH=ml/src python ml/scripts/eval/wr_matrix.py --games 200 --workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json

import tqdm
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from guandan.agents import make_agent
from guandan.cards import Rank
from guandan.rating import GlickoPlayer, glicko2_update
from _eval_worker import play_n_games

DEFAULT_AGENTS = [
    "random", "greedy", "heuristic", "strategic",
    "xingdream", "lalala", "liuzha", "hulalala",
    "yaoji", "jidan", "ez", "wjsd",
]

MatchupResult = dict[str, Any]


def run_matchup(agent_a: Any, agent_b: Any, n_games: int, label: str = "", log_every: int = 0) -> MatchupResult:
    """Run n_games with agent_a on seats {0,2} vs agent_b on seats {1,3}."""
    result = play_n_games(agent_a, agent_b, n_games, label=label, progress=log_every)
    return {**result, "injected": False}


def _matchup_worker(
    task: tuple[str, str, int, Rank, str | None, int],
) -> tuple[str, str, MatchupResult]:
    """Top-level worker for ProcessPoolExecutor; constructs agents in-process."""
    a_name, b_name, n_games, level_rank, checkpoint, log_every = task
    label = f"{a_name} vs {b_name}"
    print(f"→ {label}", flush=True)
    agent_a = make_agent(a_name, level_rank=level_rank, checkpoint=checkpoint)
    agent_b = make_agent(b_name, level_rank=level_rank, checkpoint=checkpoint)
    return a_name, b_name, run_matchup(agent_a, agent_b, n_games, label=label, log_every=log_every)


def run_all_matchups(
    live_names: list[str],
    n_games: int,
    level_rank: Rank,
    n_workers: int,
    checkpoint: str | None = None,
    preloaded: dict[str, dict[str, MatchupResult]] | None = None,
    log_every: int = 0,
) -> dict[str, dict[str, MatchupResult]]:
    """Run all directed matchups in round-robin order, returning a nested result dict.

    Matchups already present in `preloaded` are skipped.
    """
    preloaded = preloaded or {}
    tasks: list[tuple[str, str, int, Rank, str | None, int]] = [
        (a, b, n_games, level_rank, checkpoint, log_every)
        for a in live_names
        for b in live_names
        if a != b and b not in preloaded.get(a, {})
    ]
    n_matchups = len(tasks)
    matrix: dict[str, dict[str, MatchupResult]] = {n: {} for n in live_names}
    t0 = time.monotonic()

    bar = tqdm.tqdm(total=n_matchups, unit="matchup", dynamic_ncols=True)

    if n_workers == 1:
        for task in tasks:
            a_name, b_name, result = _matchup_worker(task)
            matrix[a_name][b_name] = result
            elapsed = time.monotonic() - t0
            bar.set_postfix_str(f"{a_name} vs {b_name} WR={result['winrate']:.1%} ({elapsed:.0f}s)")
            bar.update(1)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_matchup_worker, task): task for task in tasks}
            for future in concurrent.futures.as_completed(futures):
                a_name, b_name, result = future.result()
                matrix[a_name][b_name] = result
                elapsed = time.monotonic() - t0
                bar.set_postfix_str(f"{a_name} vs {b_name} WR={result['winrate']:.1%} ({elapsed:.0f}s)")
                bar.update(1)

    bar.close()

    return matrix


def parse_inject(inject_strs: list[str]) -> dict[str, dict[str, MatchupResult]]:
    """Parse --inject entries into matrix-compatible dicts.

    Each entry: "name,opponent,wins,n_games". The symmetric reverse is auto-added.
    """
    injected: dict[str, dict[str, MatchupResult]] = {}
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
    matrix: dict[str, dict[str, MatchupResult]],
    agent_names: list[str],
    n_passes: int = 30,
) -> dict[str, GlickoPlayer]:
    """Iteratively derive Glicko-2 ratings from the win-rate matrix."""
    players = {name: GlickoPlayer(name=name) for name in agent_names}

    for _ in range(n_passes):
        new_players: dict[str, GlickoPlayer] = {}
        for name in agent_names:
            opponents = []
            outcomes = []
            for opp_name in agent_names:
                if opp_name == name:
                    continue
                data = matrix.get(name, {}).get(opp_name)
                if data is None:
                    continue  # injected agents may have partial matchup data
                opponents.append(players[opp_name])
                outcomes.append(data["wins"] / data["n_games"])
            new_players[name] = glicko2_update(players[name], opponents, outcomes)
            new_players[name].name = name
        players = new_players

    return players


def print_matrix(
    matrix: dict[str, dict[str, MatchupResult]], agent_names: list[str]
) -> None:
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
                row += f"{wr:>{col_w - 1}.1%}{inj}"
        print(row)
    print("  (* = injected from prior eval, not re-run)")
    print()


def print_ratings(
    ratings: dict[str, GlickoPlayer], agent_names: list[str]
) -> None:
    sorted_names = sorted(agent_names, key=lambda n: ratings[n].rating, reverse=True)
    print("Derived Glicko-2 Ratings")
    print("=" * 40)
    print(f"{'Bot':>14}  {'Rating':>7}  {'RD':>5}")
    print("-" * 40)
    for name in sorted_names:
        p = ratings[name]
        print(f"{name:>14}  {p.rating:>7.0f}  {p.rd:>5.0f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Round-robin win rate matrix for all bots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--games", type=int, default=200,
        help="Games per directed matchup",
    )
    parser.add_argument(
        "--agents", type=str, default=None,
        help="Comma-separated agent names (default: all rule-based)",
    )
    parser.add_argument(
        "--workers", type=int, default=multiprocessing.cpu_count(),
        help="Parallel worker processes (1 = serial)",
    )
    parser.add_argument(
        "--inject", type=str, action="append", default=[],
        help=(
            "Pre-known result: 'name,opponent,wins,n_games'. "
            "Agent is added to Glicko without re-running games. "
            "Reverse is auto-inferred. Can be repeated."
        ),
    )
    parser.add_argument(
        "--output", type=str, default="ml/runs/wr_matrix",
        help="Output directory",
    )
    parser.add_argument(
        "--rating-passes", type=int, default=30,
        help="Glicko-2 convergence passes",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a DART checkpoint .pt file. Adds 'dart' to the agent list.",
    )
    parser.add_argument(
        "--load-matrix", type=str, default=None,
        help="Path to a prior results JSON. Existing matchups are reused; only missing pairs are run.",
    )
    parser.add_argument(
        "--log-every", type=int, default=0,
        help="Print per-worker progress every N games within a matchup (0 = off).",
    )
    args = parser.parse_args()

    agent_names: list[str] = args.agents.split(",") if args.agents else DEFAULT_AGENTS[:]
    if args.checkpoint and "dart" not in agent_names:
        agent_names.insert(0, "dart")

    injected_matrix = parse_inject(args.inject)
    injected_only: set[str] = set()
    for inj_name in injected_matrix:
        if inj_name not in agent_names:
            agent_names.append(inj_name)
            injected_only.add(inj_name)

    # Seed matrix from a prior results file; existing pairs won't be re-run.
    # Only matchups between agents already in agent_names are loaded — others are ignored.
    preloaded: dict[str, dict[str, MatchupResult]] = {}
    if args.load_matrix:
        prior = json.loads(Path(args.load_matrix).read_text())
        agent_set = set(agent_names)
        for a, row in prior["matrix"].items():
            if a not in agent_set:
                continue
            for b, result in row.items():
                if b not in agent_set:
                    continue
                preloaded.setdefault(a, {})[b] = {**result, "injected": True}
        print(f"Loaded {args.load_matrix} — {sum(len(v) for v in preloaded.values())} preloaded matchups")

    level_rank = Rank.TWO
    live_names = [n for n in agent_names if n not in injected_only]
    n_new = sum(
        1 for a in live_names for b in live_names
        if a != b and b not in preloaded.get(a, {})
    )
    total_games = n_new * args.games

    print(
        f"Round-robin: {len(live_names)} live agents, {n_new} new matchups, "
        f"{total_games} total games @ {args.games}/matchup  (workers={args.workers})"
    )
    if injected_only:
        print(f"Injected (no games run): {', '.join(sorted(injected_only))}")
    print(f"Agents: {', '.join(live_names)}")

    matrix: dict[str, dict[str, MatchupResult]] = {a: {} for a in agent_names}
    for a, row in injected_matrix.items():
        matrix.setdefault(a, {}).update(row)
    for a, row in preloaded.items():
        matrix.setdefault(a, {}).update(row)

    t_start = time.monotonic()
    live_matrix = run_all_matchups(
        live_names, args.games, level_rank, args.workers, args.checkpoint, preloaded, args.log_every
    )
    elapsed = time.monotonic() - t_start

    for a, row in live_matrix.items():
        matrix[a].update(row)

    print(f"\nTotal: {elapsed:.0f}s ({elapsed / 60:.1f}min)")

    print_matrix(matrix, agent_names)

    ratings = derive_ratings(matrix, agent_names, n_passes=args.rating_passes)
    print_ratings(ratings, agent_names)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "agents": agent_names,
        "games_per_matchup": args.games,
        "total_games": total_games,
        "total_time_s": round(elapsed, 1),
        "matrix": {
            a: {b: matrix[a][b] for b in agent_names if b != a and b in matrix.get(a, {})}
            for a in agent_names
        },
        "ratings": {name: ratings[name].to_dict() for name in agent_names},
    }
    out_file = out_dir / f"results_{time.strftime('%m%d_%H%M')}.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {out_file}")

    elo_path = Path(__file__).resolve().parents[2] / "src" / "guandan" / "elos.json"
    elo_export = {name: round(ratings[name].rating) for name in agent_names}
    elo_path.write_text(json.dumps(elo_export, indent=2) + "\n")
    print(f"ELOs exported to {elo_path}")


if __name__ == "__main__":
    main()
