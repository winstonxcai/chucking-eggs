"""Full ladder evaluation: RL agent vs all 5 baseline opponents.

Loads a checkpoint and evaluates against Random, Greedy, Heuristic, Strategic,
and optionally MonteCarloBot. Prints a comparison table with deltas vs known
Day 2 (flat MLP, no history) baselines.

Usage:
    python scripts/ladder.py --checkpoint runs/20260313_143000/model_final.pt
    python scripts/ladder.py --checkpoint runs/20260313_143000/model_final.pt --skip-mc
    python scripts/ladder.py --checkpoint runs/20260313_143000/model_final.pt --mc-sims 50 --mc-workers 8

Expected runtime (M1 Pro, 8 cores):
    Without MC:  ~40 min
    With MC:     ~80-100 min
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from guandan.agents import (
    GreedyBot,
    HeuristicBot,
    MonteCarloBot,
    RandomBot,
    RLAgentLSTM,
    StrategicBot,
)
from guandan.cards import Rank
from guandan.game import GuanDanEnv
from guandan.training.q_network import QNetworkLSTM, get_device

# ─── Day 2 baselines (flat MLP, no history) ──────────────────────────────────
# These are the known results from Day 2 training to use as the comparison point.
# Update these constants as you establish new baselines.
DAY2_BASELINES = {
    "Random":      0.92,
    "Greedy":      0.82,
    "Heuristic":   0.59,
    "Strategic":   0.50,
    "MonteCarlo":  0.42,
}


def eval_vs_opponent(rl_agent, opponent, name: str, n_games: int) -> dict:
    """Evaluate RL agent (team 0,2) vs opponent (team 1,3). Returns stats dict."""
    env = GuanDanEnv()
    wins = 0
    total_r = 0.0
    finish_12 = 0

    for _ in range(n_games):
        env.reset()
        while not env.done:
            p = env.current_player
            if p in (0, 2):
                move = rl_agent.act(env, p)
            else:
                move = opponent.act(env, p)
            env.step(move)

        rewards = env.get_rewards()
        team_r = rewards[0] + rewards[2]
        if team_r > 0:
            wins += 1
        total_r += team_r

        fo = env.finish_order
        if fo[0] in (0, 2) and fo[1] in (0, 2):
            finish_12 += 1

    return {
        "name": name,
        "wins": wins,
        "n_games": n_games,
        "winrate": wins / n_games,
        "avg_reward": total_r / n_games,
        "finish_12": finish_12,
    }


def print_table(results: list[dict]) -> None:
    """Print formatted results table with Day 2 deltas."""
    header = f"{'Opponent':>12} | {'Games':>5} | {'Wins':>5} | {'WR':>6} | {'Avg R':>7} | Δ vs Day2"
    sep = "─" * len(header)
    print()
    print(header)
    print(sep)
    for r in results:
        name = r["name"]
        baseline = DAY2_BASELINES.get(name)
        if baseline is not None:
            delta = r["winrate"] - baseline
            delta_str = f"{delta:+.1%}  (was {baseline:.0%})"
        else:
            delta_str = "  (no baseline)"
        mc_note = "*" if name == "MonteCarlo" else " "
        print(
            f"{name+mc_note:>12} | {r['n_games']:>5} | {r['wins']:>5} | "
            f"{r['winrate']:>5.1%} | {r['avg_reward']:>+6.2f} | {delta_str}"
        )
    if any(r["name"] == "MonteCarlo" for r in results):
        print("  * MC results are noisier (fewer games)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full ladder eval: RL agent vs all baseline opponents"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--skip-mc", action="store_true",
                        help="Skip MonteCarloBot eval (saves ~40-60 min)")
    parser.add_argument("--mc-sims", type=int, default=30,
                        help="MC rollouts per candidate (default: 30)")
    parser.add_argument("--mc-workers", type=int, default=8,
                        help="MC parallel workers (default: 8)")
    parser.add_argument("--mc-games", type=int, default=50,
                        help="Games vs MC (default: 50)")
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--mlp-hidden", type=int, default=512)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load checkpoint
    q_lead = QNetworkLSTM(lstm_hidden=args.lstm_hidden, hidden=args.mlp_hidden).to(device)
    q_follow = QNetworkLSTM(lstm_hidden=args.lstm_hidden, hidden=args.mlp_hidden).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    q_lead.load_state_dict(ckpt["lead_state_dict"], strict=False)
    q_follow.load_state_dict(ckpt["follow_state_dict"], strict=False)
    q_lead.eval()
    q_follow.eval()

    level_rank = Rank.TWO
    rl = RLAgentLSTM(q_lead, q_follow, device, level_rank)

    episode_str = f" (ep {ckpt['episode']})" if "episode" in ckpt else ""
    print(f"\n=== LADDER EVAL: {Path(args.checkpoint).name}{episode_str} ===")

    # Main ladder opponents
    opponents = [
        ("Random",    RandomBot(),                   200),
        ("Greedy",    GreedyBot(level_rank),          200),
        ("Heuristic", HeuristicBot(level_rank),       300),
        ("Strategic", StrategicBot(level_rank),       300),
    ]

    results = []
    for name, opp, n_games in opponents:
        t0 = time.time()
        print(f"  Evaluating vs {name} ({n_games} games)...", end="", flush=True)
        r = eval_vs_opponent(rl, opp, name, n_games)
        elapsed = time.time() - t0
        print(f" {r['winrate']:.1%}  ({elapsed:.0f}s)")
        results.append(r)

    # MC eval
    if not args.skip_mc:
        mc = MonteCarloBot(
            n_sims=args.mc_sims,
            n_workers=args.mc_workers,
            level_rank=level_rank,
        )
        print(
            f"  Evaluating vs MonteCarlo ({args.mc_games} games, "
            f"n_sims={args.mc_sims}, n_workers={args.mc_workers})..."
        )
        mc_results = {"name": "MonteCarlo", "wins": 0, "n_games": args.mc_games,
                      "avg_reward": 0.0, "finish_12": 0}
        total_r = 0.0
        env = GuanDanEnv()
        for i in range(args.mc_games):
            env.reset()
            while not env.done:
                p = env.current_player
                if p in (0, 2):
                    move = rl.act(env, p)
                else:
                    move = mc.act(env, p)
                env.step(move)
            rewards = env.get_rewards()
            team_r = rewards[0] + rewards[2]
            if team_r > 0:
                mc_results["wins"] += 1
            total_r += team_r
            fo = env.finish_order
            if fo[0] in (0, 2) and fo[1] in (0, 2):
                mc_results["finish_12"] += 1
            if (i + 1) % 10 == 0:
                wr = mc_results["wins"] / (i + 1)
                print(f"    {i+1}/{args.mc_games} — running WR: {wr:.1%}")
        mc_results["winrate"] = mc_results["wins"] / args.mc_games
        mc_results["avg_reward"] = total_r / args.mc_games
        results.append(mc_results)
    else:
        print("  [Skipping MC eval — use without --skip-mc for full ladder]")

    print_table(results)
    print()


if __name__ == "__main__":
    main()
