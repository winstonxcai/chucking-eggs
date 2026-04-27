"""Paired evaluation for the pvguan PV-AC vs PV-PTIE ablation.

Two bots play every deal from both team orientations, with all 4 seat rotations
per deal so each bot experiences every physical seat equally.

Usage:
    # Primary ablation comparison (PV-PTIE vs PV-AC)
    PYTHONPATH=ml/src python ml/scripts/eval/paired_eval.py \\
        --bot-a ml/checkpoints/pvguan_ptie_seed0_final.pt \\
        --bot-b ml/checkpoints/pvguan_pv_seed0_final.pt \\
        --deals 1000 --seed-range sealed \\
        --out ml/runs/final_sealed_eval/ptie0_vs_pv0.json

    # Validation eval (against Jidan)
    PYTHONPATH=ml/src python ml/scripts/eval/paired_eval.py \\
        --bot-a ml/checkpoints/pvguan_pv_seed0_best.pt \\
        --opponent jidan \\
        --deals 200 --seed-range validation \\
        --out ml/runs/pvguan_pv_seed0/val_vs_jidan.json

Seed ranges (non-overlapping):
    distill   : [10_000_000, 12_000_000)
    training  : [20_000_000, ∞)
    validation: [30_000_000, 30_200_000)   200 deals
    budget    : [40_000_000, 40_200_000)   200 deals
    sealed    : [50_000_000, 51_000_000)   1000 deals
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from guandan.agents import JidanBot, make_agent
from guandan.cards import Rank
from guandan.game import GuanDanEnv
from guandan.pvguan.agent import PVGuanBot


# ─── Seed partitions ──────────────────────────────────────────────────────────

SEED_RANGES = {
    "distill":    (10_000_000, 12_000_000),
    "training":   (20_000_000, 100_000_000),
    "validation": (30_000_000, 30_200_000),
    "budget":     (40_000_000, 40_200_000),
    "sealed":     (50_000_000, 51_000_000),
}


def get_deal_seeds(range_name: str, n: int) -> list[int]:
    """Return n evenly-spaced seeds from the named range."""
    lo, hi = SEED_RANGES[range_name]
    if n > (hi - lo):
        raise ValueError(f"Requested {n} seeds but range '{range_name}' only has {hi - lo}")
    step = (hi - lo) // n
    return [lo + i * step for i in range(n)]


# ─── Agent builder ────────────────────────────────────────────────────────────

def _build_bot(spec: str, level_rank: int) -> object:
    """Build a bot from a spec string: checkpoint path or well-known name."""
    p = Path(spec)
    if p.exists() and p.suffix == ".pt":
        return PVGuanBot(p, level_rank=level_rank, sample=False)
    return make_agent(spec, level_rank=level_rank)


# ─── One deal, 4 seat rotations ───────────────────────────────────────────────

def _run_rotation(
    deal_seed: int,
    level_rank: int,
    bots: tuple,   # (bot_a, bot_b) — bot_a on seats {team_a_seats}
    team_a_seats: tuple[int, int],
) -> dict:
    """Run one hand with fixed deal_seed and the given seat assignment.

    bot_a occupies team_a_seats, bot_b occupies the other two seats.
    Returns outcome: {promotion_diff, finish_order, win}.
    """
    bot_a, bot_b = bots
    team_b_seats = tuple(s for s in (0, 1, 2, 3) if s not in team_a_seats)

    env = GuanDanEnv(level_rank=level_rank)
    env.reset(seed=deal_seed)

    while not env.done:
        p = env.current_player
        if p in team_a_seats:
            move = bot_a.act(env, p)
        else:
            move = bot_b.act(env, p)
        env.step(move)

    rewards = env.get_rewards()
    promo_diff = rewards[team_a_seats[0]]
    return {
        "promotion_diff": float(promo_diff),
        "finish_order":   list(env.finish_order),
        "win":            promo_diff > 0,
        "loss":           promo_diff < 0,
        "tie":            promo_diff == 0,
        "team_a_seats":   list(team_a_seats),
        "deal_seed":      deal_seed,
        "level_rank":     level_rank,
    }


def _finish_split(outcomes: list[dict]) -> dict:
    """1-2 / 1-3 / 1-4 finish split, conditional on wins."""
    wins = [o for o in outcomes if o["win"]]
    if not wins:
        return {"1_2": 0.0, "1_3": 0.0, "1_4": 0.0}
    counts = {"1_2": 0, "1_3": 0, "1_4": 0}
    for o in wins:
        fo = o["finish_order"]
        team_a = set(o["team_a_seats"])
        positions = [fo.index(s) for s in team_a]
        p1, p2 = sorted(positions)
        if p1 == 0 and p2 == 1:
            counts["1_2"] += 1
        elif p1 == 0 and p2 == 2:
            counts["1_3"] += 1
        else:
            counts["1_4"] += 1
    total = len(wins)
    return {k: v / total for k, v in counts.items()}


# ─── Block bootstrap ──────────────────────────────────────────────────────────

def _bootstrap_ci(
    deal_blocks: dict[int, list[float]],
    n_resample: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Block bootstrap by deal seed.

    deal_blocks: {seed: [promo_diff per rotation]}
    Returns (mean, lower_ci, upper_ci).
    """
    seeds = list(deal_blocks.keys())
    rng = random.Random(12345)
    sample_means = []
    for _ in range(n_resample):
        chosen = [rng.choice(seeds) for _ in seeds]
        vals = [v for s in chosen for v in deal_blocks[s]]
        sample_means.append(float(np.mean(vals)))
    sample_means.sort()
    lo = sample_means[int(alpha / 2 * n_resample)]
    hi = sample_means[int((1 - alpha / 2) * n_resample)]
    all_vals = [v for vals in deal_blocks.values() for v in vals]
    return float(np.mean(all_vals)), lo, hi


# ─── Main eval loop ───────────────────────────────────────────────────────────

def run_paired_eval(
    bot_a,
    bot_b,
    deal_seeds: list[int],
    level_ranks: list[int] | None = None,
    n_bootstrap: int = 1000,
) -> dict:
    """Run all deals × 4 rotations. Returns full results dict."""
    # Four seat-rotation assignments (team_a occupies each pair of opposite seats)
    ROTATIONS: list[tuple[int, int]] = [(0, 2), (1, 3), (0, 2), (1, 3)]
    # For symmetric reporting, rotate through all 4 seat combos
    # Actually: 2 team-orientation options × 2 home/away = 4 rotations
    ROT_TEAM_A = [(0, 2), (0, 2), (1, 3), (1, 3)]

    all_outcomes: list[dict] = []
    deal_blocks: dict[int, list[float]] = {}

    _level_ranks = level_ranks or list(range(Rank.TWO, Rank.ACE + 1))

    for i, seed in enumerate(deal_seeds):
        # Sample a level rank deterministically from seed
        lvl = _level_ranks[seed % len(_level_ranks)]
        block_diffs = []
        for team_a_seats in ROT_TEAM_A:
            outcome = _run_rotation(seed, lvl, (bot_a, bot_b), team_a_seats)
            all_outcomes.append(outcome)
            block_diffs.append(outcome["promotion_diff"])
        deal_blocks[seed] = block_diffs

    diffs = [o["promotion_diff"] for o in all_outcomes]
    mean_diff = float(np.mean(diffs))
    win_rate = float(np.mean([o["win"] for o in all_outcomes]))
    tie_rate = float(np.mean([o["tie"] for o in all_outcomes]))

    mean_bs, ci_lo, ci_hi = _bootstrap_ci(deal_blocks, n_resample=n_bootstrap)
    finish = _finish_split(all_outcomes)

    return {
        "n_deals":             len(deal_seeds),
        "n_hands":             len(all_outcomes),
        "promotion_diff_mean": mean_diff,
        "promotion_diff_ci":   [ci_lo, ci_hi],
        "win_rate":            win_rate,
        "tie_rate":            tie_rate,
        "loss_rate":           1.0 - win_rate - tie_rate,
        "finish_split":        finish,
        "outcomes":            all_outcomes,   # full per-hand log
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _print_summary(label: str, results: dict) -> None:
    d = results["promotion_diff_mean"]
    ci = results["promotion_diff_ci"]
    wr = results["win_rate"]
    fs = results["finish_split"]
    print(f"\n{label}")
    print(f"  promotion_diff:  {d:+.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"  win_rate:        {wr:.3f}  (tie={results['tie_rate']:.3f})")
    print(f"  finish split (cond. on win): "
          f"1-2={fs['1_2']:.2%}  1-3={fs['1_3']:.2%}  1-4={fs['1_4']:.2%}")
    print(f"  n_hands: {results['n_hands']}")


def main() -> None:
    parser = argparse.ArgumentParser("paired_eval")
    parser.add_argument("--bot-a",     required=True,
                        help="Checkpoint path or bot name for team A")
    parser.add_argument("--bot-b",     default=None,
                        help="Checkpoint path or bot name for team B (default: jidan)")
    parser.add_argument("--opponent",  default=None,
                        help="Alias for --bot-b (for single-bot eval usage)")
    parser.add_argument("--deals",     type=int, default=200)
    parser.add_argument("--seed-range", default="validation",
                        choices=list(SEED_RANGES.keys()))
    parser.add_argument("--level-rank", type=int, default=None,
                        help="Fix level rank (default: cycle through all 13)")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    bot_b_spec = args.bot_b or args.opponent or "jidan"
    level_rank = args.level_rank or Rank.TWO

    bot_a = _build_bot(args.bot_a, level_rank)
    bot_b = _build_bot(bot_b_spec, level_rank)

    deal_seeds = get_deal_seeds(args.seed_range, args.deals)
    level_ranks = [args.level_rank] if args.level_rank else None

    results = run_paired_eval(
        bot_a, bot_b, deal_seeds,
        level_ranks=level_ranks,
        n_bootstrap=args.n_bootstrap,
    )
    results["bot_a"] = args.bot_a
    results["bot_b"] = bot_b_spec
    results["seed_range"] = args.seed_range

    if not args.quiet:
        _print_summary(f"{args.bot_a}  vs  {bot_b_spec}", results)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Don't dump per-hand outcomes to JSON by default (too large)
        summary = {k: v for k, v in results.items() if k != "outcomes"}
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
