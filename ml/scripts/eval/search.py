"""Compare RL agent with and without inference-time search.

Usage:
    PYTHONPATH=src python scripts/eval_search.py \
        --checkpoint checkpoints/selfplay_best.pt \
        --games 500 --n-worlds 20 --depth 3 --prune-top-k 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from guandan.agents import make_agent
from guandan.agents.rl_agent import RLAgentLSTM
from guandan.game import GuanDanEnv
from guandan.search.search_agent import SearchAgent
from guandan.training.q_network import QNetworkLSTM, get_device, load_compat


def load_checkpoint(path: str, device: torch.device, use_gnn: bool = False):
    """Load Q-networks from checkpoint. Auto-detects GNN from weights."""
    ckpt = torch.load(path, map_location=device, weights_only=True)

    lstm_hidden = ckpt.get("lstm_hidden", 256)
    mlp_hidden = ckpt.get("mlp_hidden", 1024)

    # Auto-detect GNN from checkpoint weight shapes
    lead_key = "lead_state_dict" if "lead_state_dict" in ckpt else "lead"
    follow_key = "follow_state_dict" if "follow_state_dict" in ckpt else "follow"
    sd = ckpt[lead_key]
    if "mlp.0.weight" in sd and sd["mlp.0.weight"].shape[1] > 833:
        use_gnn = True
        print("  Auto-detected GNN checkpoint")

    q_lead = QNetworkLSTM(
        lstm_hidden=lstm_hidden, hidden=mlp_hidden, use_gnn=use_gnn,
    ).to(device)
    q_follow = QNetworkLSTM(
        lstm_hidden=lstm_hidden, hidden=mlp_hidden, use_gnn=use_gnn,
    ).to(device)

    load_compat(q_lead, ckpt[lead_key])
    load_compat(q_follow, ckpt[follow_key])
    q_lead.eval()
    q_follow.eval()

    return q_lead, q_follow, ckpt


def run_games(agent, opponent, n_games: int, level_rank: int,
              seed: int | None = None) -> dict:
    """Run n_games and return results dict."""
    if seed is not None:
        import random as _random
        _random.seed(seed)
    env = GuanDanEnv(level_rank)
    wins = 0
    total_reward = 0.0
    finish_12 = finish_13 = finish_14 = 0
    finish_23 = finish_24 = finish_34 = 0

    t0 = time.time()
    for g in range(n_games):
        env.reset()
        while not env.done:
            p = env.current_player
            if p in (0, 2):
                move = agent.act(env, p)
            else:
                move = opponent.act(env, p)
            env.step(move)

        rewards = env.get_rewards()
        r = rewards[0]
        total_reward += r
        if r > 0:
            wins += 1

        fo = env.finish_order
        tp = tuple(sorted([fo.index(0), fo.index(2)]))
        if tp == (0, 1): finish_12 += 1
        elif tp == (0, 2): finish_13 += 1
        elif tp == (0, 3): finish_14 += 1
        elif tp == (1, 2): finish_23 += 1
        elif tp == (1, 3): finish_24 += 1
        elif tp == (2, 3): finish_34 += 1

        if (g + 1) % max(1, n_games // 10) == 0:
            elapsed = time.time() - t0
            wr = wins / (g + 1)
            print(f"  [{g+1}/{n_games}] WR={wr:.1%} ({elapsed:.1f}s)", flush=True)

    elapsed = time.time() - t0
    n = n_games
    return {
        "wins": wins, "winrate": wins / n, "avg_reward": total_reward / n,
        "finish_12": finish_12, "finish_13": finish_13, "finish_14": finish_14,
        "finish_23": finish_23, "finish_24": finish_24, "finish_34": finish_34,
        "elapsed": elapsed, "games": n,
    }


def print_results(name: str, r: dict) -> None:
    n = r["games"]
    wr = r["winrate"]
    level_eff = (3*r["finish_12"] + 2*r["finish_13"] + r["finish_14"]
                 - r["finish_23"] - 2*r["finish_24"] - 3*r["finish_34"]) / n
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Win rate:        {wr:.1%}  ({r['wins']}/{n})")
    print(f"  Avg reward:      {r['avg_reward']:+.3f}")
    print(f"  Net levels/game: {level_eff:+.3f}")
    print(f"  --- Wins ---")
    print(f"  1-2: {r['finish_12']/n:.1%}  1-3: {r['finish_13']/n:.1%}  1-4: {r['finish_14']/n:.1%}")
    print(f"  --- Losses ---")
    print(f"  2-3: {r['finish_23']/n:.1%}  2-4: {r['finish_24']/n:.1%}  3-4: {r['finish_34']/n:.1%}")
    print(f"  Time: {r['elapsed']:.1f}s ({r['elapsed']/n:.2f}s/game)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate search vs no-search")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--opponent", type=str, default="strategic")
    parser.add_argument("--n-worlds", type=int, default=20)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--prune-top-k", type=int, default=5)
    parser.add_argument("--use-gnn", action="store_true")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU (faster for search with small batches)")
    parser.add_argument("--search-only", action="store_true",
                        help="Skip baseline, only run search agent")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Skip search, only run baseline agent")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else get_device()
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    q_lead, q_follow, ckpt = load_checkpoint(
        args.checkpoint, device, use_gnn=args.use_gnn,
    )
    level_rank = ckpt.get("level_rank", 2)

    if "episode" in ckpt:
        print(f"Trained for {ckpt['episode']} episodes")

    opp = make_agent(args.opponent, level_rank)
    print(f"Opponent: {args.opponent}")
    print(f"Games: {args.games}")

    # Baseline: no search
    if not args.search_only:
        print(f"\n--- Baseline (argmax Q) ---")
        baseline_agent = RLAgentLSTM(q_lead, q_follow, device, level_rank)
        baseline = run_games(baseline_agent, opp, args.games, level_rank,
                             seed=args.seed)
        print_results(f"Baseline vs {args.opponent}", baseline)

    # Search — use opponent bot for rollouts (hybrid: Q for our team, bot for opponents)
    if not args.baseline_only:
        rollout_opp = make_agent(args.opponent, level_rank)
        print(f"\n--- Search (worlds={args.n_worlds}, depth={args.depth}, "
              f"top_k={args.prune_top_k}, rollout_opp={args.opponent}) ---")
        search_agent = SearchAgent(
            q_lead, q_follow, device, level_rank,
            n_worlds=args.n_worlds, depth=args.depth,
            prune_top_k=args.prune_top_k,
            opp_bot=rollout_opp,
        )
        search = run_games(search_agent, opp, args.games, level_rank,
                           seed=args.seed)
        print_results(
            f"Search (w={args.n_worlds},d={args.depth},k={args.prune_top_k}) "
            f"vs {args.opponent}",
            search,
        )

    # Comparison
    if not args.search_only and not args.baseline_only:
        delta = search["winrate"] - baseline["winrate"]
        print(f"\n{'='*50}")
        print(f"  Search improvement: {delta:+.1%}")
        print(f"  Baseline: {baseline['winrate']:.1%} → Search: {search['winrate']:.1%}")
        print(f"  Speed: {baseline['elapsed']/baseline['games']:.3f}s → "
              f"{search['elapsed']/search['games']:.3f}s per game")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
