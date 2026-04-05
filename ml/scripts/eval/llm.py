"""Evaluate LLMBot (GPT-5.4 Nano) against competition bots.

Usage:
    PYTHONPATH=src python scripts/eval_llm.py --games 20 --opponents jidan
    PYTHONPATH=src python scripts/eval_llm.py --games 200 --opponents jidan,yaoji
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from guandan.agents import make_agent
from guandan.agents.llm_bot import LLMBot
from guandan.cards import Rank
from guandan.game import GuanDanEnv

# ── GPT-5.4 Nano pricing (USD per 1M tokens) ──────────────────────────────────
_INPUT_PRICE = 0.20
_CACHED_PRICE = 0.02
_OUTPUT_PRICE = 1.25


def eval_vs_opponent(
    llm_agent: LLMBot,
    opponent,
    name: str,
    n_games: int,
    level_rank: int = Rank.TWO,
) -> dict:
    """Run LLM agent (seats 0,2) vs opponent (seats 1,3)."""
    env = GuanDanEnv(level_rank)
    wins = 0
    finish_12 = 0
    t0 = time.time()

    for g in range(n_games):
        env.reset()
        while not env.done:
            p = env.current_player
            if p in (0, 2):
                move = llm_agent.act(env, p)
            else:
                move = opponent.act(env, p)
            env.step(move)

        rewards = env.get_rewards()
        if rewards[0] + rewards[2] > 0:
            wins += 1

        fo = env.finish_order
        if fo[0] in (0, 2) and fo[1] in (0, 2):
            finish_12 += 1

        if (g + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (g + 1) / elapsed
            remaining = (n_games - g - 1) / max(rate, 0.001)
            print(
                f"  [{g+1}/{n_games}] WR={wins/(g+1):.1%}  "
                f"{rate:.1f}g/s  ETA {remaining/60:.0f}min"
            )

    elapsed = time.time() - t0
    return {
        "name": name,
        "n_games": n_games,
        "wins": wins,
        "winrate": wins / n_games,
        "finish_12": finish_12,
        "elapsed_s": elapsed,
    }


def estimate_cost(total_tokens: int) -> float:
    """Rough cost estimate assuming 80% cached input, 10% output."""
    # Approximation: 70% cached input, 20% uncached input, 10% output
    cached = total_tokens * 0.70
    uncached = total_tokens * 0.20
    output = total_tokens * 0.10
    return (cached * _CACHED_PRICE + uncached * _INPUT_PRICE + output * _OUTPUT_PRICE) / 1e6


def print_results(results: list[dict], stats: dict) -> None:
    print()
    print(f"{'Opponent':>12} | {'Games':>5} | {'WR':>6} | {'1st+2nd':>7} | {'Time':>6}")
    print("─" * 52)
    for r in results:
        print(
            f"{r['name']:>12} | {r['n_games']:>5} | {r['winrate']:>5.1%} | "
            f"{r['finish_12']:>7} | {r['elapsed_s']:>5.0f}s"
        )

    print()
    print("─── LLM Stats ───")
    print(f"  Total LLM calls : {stats['total_calls']:,}")
    print(f"  Total tokens    : {stats['total_tokens']:,}")
    print(f"  Fallback rate   : {stats['fallback_rate']:.1%}")
    print(f"  Est. cost       : ${estimate_cost(stats['total_tokens']):.3f}")
    print(f"  ToM calls       : {stats.get('tom_calls', 0):,}  ({stats.get('tom_call_rate', 0):.1%})")
    print(f"  Intent distribution:")
    for intent, count in stats["intent_counts"].items():
        pct = count / max(1, sum(stats["intent_counts"].values()))
        print(f"    {intent:>10}: {count:4d}  ({pct:.0%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLMBot vs competition bots")
    parser.add_argument(
        "--opponents", type=str, default="jidan,yaoji",
        help="Comma-separated opponent names (default: jidan,yaoji)",
    )
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--model", type=str, default=LLMBot.DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--tom-level", type=int, default=1, choices=[0, 1, 2],
                        help="ToM level: 0=off, 1=1st-order, 2=2nd-order (default: 1)")
    parser.add_argument("--tom-threshold", type=int, default=15,
                        help="Activate ToM when min opp cards ≤ N (default: 15)")
    parser.add_argument(
        "--run-name", type=str,
        default=datetime.now().strftime("llm_%m%d_%H%M"),
    )
    args = parser.parse_args()

    level_rank = Rank.TWO
    opp_names = [o.strip() for o in args.opponents.split(",")]

    print(f"LLMBot eval — model={args.model} | games={args.games} | opponents={opp_names}")
    print(f"Top-K={args.top_k} | temperature={args.temperature} | ToM level={args.tom_level} threshold={args.tom_threshold}")
    print()

    llm_agent = LLMBot(
        level_rank=level_rank,
        model=args.model,
        temperature=args.temperature,
        top_k=args.top_k,
        tom_level=args.tom_level,
        tom_min_opp=args.tom_threshold,
    )

    results = []
    for opp_name in opp_names:
        opp = make_agent(opp_name, level_rank=level_rank)
        print(f"Evaluating vs {opp_name} ({args.games} games)...")
        result = eval_vs_opponent(llm_agent, opp, opp_name, args.games, level_rank)
        results.append(result)
        print(f"  → WR={result['winrate']:.1%}  1st+2nd={result['finish_12']}")

    stats = llm_agent.stats
    print_results(results, stats)

    # Save results
    run_dir = Path("ml/runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "model": args.model,
        "games": args.games,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "results": results,
        "llm_stats": stats,
        "estimated_cost_usd": estimate_cost(stats["total_tokens"]),
    }
    out_path = run_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
