"""Compare rollout PIMC scores at depth_limit=30 vs depth_limit=15.

Uses identical determinizations + identical rollout policies for both depths.
The only difference is when the rollout cuts off and the leaf-value heuristic
takes over. If signals correlate strongly, the shorter depth is a free speedup.

Usage:
    PYTHONPATH=ml/src python ml/scripts/util/compare_depth.py \\
        --checkpoint ml/checkpoints/az_gen5_mix_bestwr.pt \\
        --decisions 100 --n-det 20 --top-k 3 \\
        --depth-a 30 --depth-b 15
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from guandan.agents.partner_oracle_bot import PartnerOracleBot
from guandan.agents.partner_pimc_bot import (
    _clone_env,
    _determinize,
    _rollout_limited,
    ROLLOUT_FACTORIES,
)
from guandan.cards import Rank
from guandan.game import GuanDanEnv


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 1.0
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return 1.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1.0)
    q = np.clip(q, 1e-9, 1.0)
    return float((p * np.log(p / q)).sum())


def score_at_depths(
    env: GuanDanEnv,
    player: int,
    candidates: list,
    n_det: int,
    rollout_mix: list[str],
    rng: random.Random,
    depth_a: int,
    depth_b: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Score candidates with the SAME determinizations + rollout-policy choices
    at two different depth_limits. Returns (scores_a, scores_b, time_a, time_b).
    """
    partner = (player + 2) % 4
    dets = [_determinize(env, player, partner, rng) for _ in range(n_det)]
    # Pre-pick rollout policy per det so both depths use identical policies
    policies = [random.choice(rollout_mix) for _ in range(n_det)]

    scores_a = np.zeros(len(candidates), dtype=np.float64)
    scores_b = np.zeros(len(candidates), dtype=np.float64)

    t0 = time.time()
    for det, pol in zip(dets, policies):
        cls = ROLLOUT_FACTORIES[pol]
        agents = [cls(env.level_rank) for _ in range(4)]
        for i, move in enumerate(candidates):
            sim = _clone_env(det)
            sim.step(move)
            scores_a[i] += _rollout_limited(sim, agents, depth_a, player)
    t_a = time.time() - t0

    t0 = time.time()
    for det, pol in zip(dets, policies):
        cls = ROLLOUT_FACTORIES[pol]
        agents = [cls(env.level_rank) for _ in range(4)]
        for i, move in enumerate(candidates):
            sim = _clone_env(det)
            sim.step(move)
            scores_b[i] += _rollout_limited(sim, agents, depth_b, player)
    t_b = time.time() - t0

    return scores_a, scores_b, t_a, t_b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--decisions", type=int, default=100)
    ap.add_argument("--n-det", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--depth-a", type=int, default=30, help="Reference depth")
    ap.add_argument("--depth-b", type=int, default=15, help="Faster depth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollout-mix", default="jidan,yaoji,strategic")
    args = ap.parse_args()

    rollout_mix = [s.strip() for s in args.rollout_mix.split(",")]
    rng = random.Random(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    oracle = PartnerOracleBot(
        checkpoint_path=args.checkpoint,
        level_rank=Rank.TWO,
        use_search=False,
        top_k=args.top_k,
        device=torch.device("cpu"),
    )

    env = GuanDanEnv(level_rank=Rank.TWO)
    spearmans, kls, argmax_agree = [], [], []
    total_t_a, total_t_b = 0.0, 0.0
    rec = 0

    while rec < args.decisions:
        env.reset()
        while not env.done and rec < args.decisions:
            p = env.current_player
            legal = env.legal_moves(p)
            if len(legal) == 1:
                env.step(legal[0])
                continue
            cands = oracle._policy_top_k(env, p, legal)
            if len(cands) <= 1:
                env.step(cands[0] if cands else legal[0])
                continue

            sa, sb, ta, tb = score_at_depths(
                env, p, cands, args.n_det, rollout_mix, rng,
                args.depth_a, args.depth_b,
            )
            total_t_a += ta
            total_t_b += tb
            pi_a = _softmax(sa)
            pi_b = _softmax(sb)

            spearmans.append(_spearman(sa, sb))
            kls.append(_kl(pi_a, pi_b))
            argmax_agree.append(int(np.argmax(sa) == np.argmax(sb)))
            rec += 1
            if rec % 20 == 0:
                print(f"  {rec}/{args.decisions}  "
                      f"r={np.mean(spearmans):.3f}  "
                      f"argmax_agree={np.mean(argmax_agree):.1%}  "
                      f"KL={np.mean(kls):.3f}  "
                      f"speedup={total_t_a/max(total_t_b,1e-6):.2f}x")

            env.step(cands[int(np.argmax(sa))])

    sp = np.array(spearmans)
    print()
    print("=" * 60)
    print(f"Decisions compared:    {len(sp)}  (depth {args.depth_a} vs {args.depth_b})")
    print(f"Spearman r:            mean={sp.mean():.3f}  median={np.median(sp):.3f}  "
          f"std={sp.std():.3f}")
    print(f"  r >= 0.9:            {(sp >= 0.9).mean():.1%}")
    print(f"  r >= 0.5:            {(sp >= 0.5).mean():.1%}")
    print(f"  r <= 0:              {(sp <= 0.0).mean():.1%}")
    print(f"Argmax agreement:      {np.mean(argmax_agree):.1%}")
    print(f"Mean KL(a||b):         {np.mean(kls):.3f}")
    print(f"Total time:            depth_a={total_t_a:.1f}s  depth_b={total_t_b:.1f}s  "
          f"speedup={total_t_a/max(total_t_b,1e-6):.2f}x")
    print("=" * 60)
    if sp.mean() > 0.85 and np.mean(argmax_agree) > 0.80:
        print(f"VERDICT: depth={args.depth_b} is a faithful proxy for depth={args.depth_a}. "
              f"Free speedup.")
    elif sp.mean() > 0.5:
        print("VERDICT: Signals partially agree. Consider intermediate depth.")
    else:
        print("VERDICT: Signals diverge. Keep the longer depth.")


if __name__ == "__main__":
    main()
