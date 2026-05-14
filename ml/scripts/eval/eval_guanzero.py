"""Eval a GuanZero checkpoint against a fixed opponent (fixed-deck / paired).

For the requested ``--games`` total ``n``, we draw ``n/2`` unique decks (one per
seed) and play each deck twice: once with GuanZero on seats (0,2), once on
(1,3). Both playthroughs start from identical hands and starting player, so the
even- vs. odd-seating win rates are directly comparable on the same deals —
the standard variance-reduction trick for card-game eval.

By default games run on CPU and fan out across N worker processes — for the
small per-move batches in self-play eval, CPU dispatch beats MPS and 6 cores
gives near-linear speedup over a sequential MPS run.

Usage:
    PYTHONPATH=ml/src python ml/scripts/eval/eval_guanzero.py \\
        --checkpoint ml/runs/<run>/checkpoints/update_00200000.pt \\
        --opponent random --games 1000
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eval_worker import play_n_games


# ─── Worker (top-level so spawn can pickle it) ───────────────────────────────


def _worker(args: tuple) -> tuple[int, int, int, int]:
    """Play a slice of games. Returns (wins_even, n_even, wins_odd, n_odd)."""
    checkpoint, opponent_name, device, even_seeds, odd_seeds = args

    import torch
    # One thread per worker — let the OS run all workers in parallel without
    # internal thread contention dragging total throughput down.
    torch.set_num_threads(1)

    from guandan.agents import make_agent
    from guandan.game import GuanDanEnv
    from guandan.guanzero.agent import GuanZeroBot

    bot = GuanZeroBot.load(checkpoint, device=device)
    opp = make_agent(opponent_name)

    # even seating: bot on {0,2}, opp on {1,3}
    r_even = play_n_games(bot, opp, len(even_seeds), seeds=even_seeds)
    # odd seating: opp on {0,2}, bot on {1,3} — wins_o counts opp wins so invert
    r_odd  = play_n_games(opp, bot, len(odd_seeds),  seeds=odd_seeds)
    wins_o = r_odd["losses"]  # bot wins = opp losses
    return r_even["wins"], len(even_seeds), wins_o, len(odd_seeds)


# ─── Driver ──────────────────────────────────────────────────────────────────


def _split(seq: list, n: int) -> list[list]:
    """Split seq into n roughly-equal contiguous chunks."""
    if n <= 0:
        return [seq]
    k, r = divmod(len(seq), n)
    out = []
    i = 0
    for w in range(n):
        size = k + (1 if w < r else 0)
        out.append(seq[i : i + size])
        i += size
    return out


def run_eval(
    checkpoint: Path,
    opponent_name: str,
    n_games: int,
    device: str,
    workers: int,
    seed: int,
) -> dict:
    if n_games % 2 != 0:
        raise ValueError(f"--games must be even for paired fixed-deck eval, got {n_games}")

    # Same seed list for both seatings: each deck is played once with GuanZero
    # on (0,2) and once on (1,3). reset(seed=N) is fully deterministic, so both
    # playthroughs start from identical hands and starting player.
    half = n_games // 2
    deck_seeds = list(range(seed, seed + half))

    even_chunks = _split(deck_seeds, workers)
    odd_chunks  = _split(deck_seeds, workers)
    tasks = [
        (checkpoint, opponent_name, device, e, o)
        for e, o in zip(even_chunks, odd_chunks)
    ]

    wins_e = wins_o = 0
    n_e = n_o = 0
    ctx = mp.get_context("spawn")
    desc = f"vs {opponent_name} ({n_games} games, {workers} workers)"
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = [pool.submit(_worker, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="chunk"):
            we, ne, wo, no = f.result()
            wins_e += we; n_e += ne
            wins_o += wo; n_o += no

    out: dict = {}
    if n_e:
        wr = wins_e / n_e; se = math.sqrt(wr * (1 - wr) / n_e)
        out["even"] = {"games": n_e, "wins": wins_e, "win_rate": wr, "se": se}
    if n_o:
        wr = wins_o / n_o; se = math.sqrt(wr * (1 - wr) / n_o)
        out["odd"]  = {"games": n_o, "wins": wins_o, "win_rate": wr, "se": se}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--opponent", default="random",
                   choices=[
                       "random",
                       "greedy",
                       "heuristic",
                       "strategic",
                       "xingdream",
                       "yaoji",
                       "jidan",
                   ])
    p.add_argument("--games", type=int, default=200,
                   help="Total games (must be even). n/2 unique decks, each played twice "
                        "— once with GuanZero on (0,2), once on (1,3).")
    p.add_argument("--workers", type=int, default=6,
                   help="Worker processes (default 6, sized for M1 Pro perf cores).")
    p.add_argument("--device", default="cpu",
                   help="Torch device for the GuanZero forward pass. Default cpu — for the "
                        "small per-move batches in eval, CPU beats MPS due to dispatch latency.")
    p.add_argument("--seed", type=int, default=0, help="Base seed for env.reset.")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional path to save structured JSON results.")
    args = p.parse_args()

    result = run_eval(args.checkpoint, args.opponent, args.games, args.device,
                      args.workers, args.seed)
    print()
    total_w = total_n = 0
    for label, r in result.items():
        print(f"  {args.checkpoint.name}  vs {args.opponent}  ({label}): "
              f"{r['win_rate']:.1%} ± {r['se']:.1%}  ({r['wins']}/{r['games']})")
        total_w += r["wins"]; total_n += r["games"]
    if len(result) > 1:
        wr = total_w / total_n
        se = math.sqrt(wr * (1 - wr) / total_n)
        print(f"  {args.checkpoint.name}  vs {args.opponent}  (combined): "
              f"{wr:.1%} ± {se:.1%}  ({total_w}/{total_n})")

    if args.out is not None:
        combined_wr = total_w / total_n if total_n else 0.0
        combined_se = math.sqrt(combined_wr * (1 - combined_wr) / total_n) if total_n else 0.0
        out_data = {
            "checkpoint": str(args.checkpoint),
            "opponent": args.opponent,
            "n_games": total_n,
            "wins": total_w,
            "wr": round(combined_wr, 4),
            "se": round(combined_se, 4),
            **{f"wr_{label}": round(r["win_rate"], 4) for label, r in result.items()},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out_data, indent=2))
        print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
