"""Compare V-at-leaf vs rollout PIMC scores on identical positions.

For each decision encountered while playing self-play games:
  1. Generate N determinizations (shared between both methods)
  2. Score top-K candidates with rollouts -> pi_rollout
  3. Score top-K candidates with V-leaf  -> pi_vleaf
  4. Record per-decision: Spearman r, argmax agreement, KL divergence

Reports aggregate stats. If r > 0.85 and argmax-agreement > 80%, V-leaf is a
faithful proxy for rollouts on this checkpoint.

Usage:
    PYTHONPATH=ml/src python ml/scripts/util/compare_vleaf_rollouts.py \\
        --checkpoint ml/checkpoints/az_gen5_mix_bestwr.pt \\
        --decisions 200 --n-det 20 --top-k 3
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from guandan.agents.partner_oracle_bot import PartnerOracleBot, _reflect_env
from guandan.agents.partner_pimc_bot import (
    _clone_env,
    _determinize,
    _encode_for_value,
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
    """Spearman rank correlation. Returns 1.0 if either is constant."""
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


def score_both(
    env: GuanDanEnv,
    player: int,
    candidates: list,
    value_net,
    n_det: int,
    rollout_mix: list[str],
    rng: random.Random,
    depth_limit: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Score candidates with both methods on the SAME determinizations."""
    partner = (player + 2) % 4
    dets = [_determinize(env, player, partner, rng) for _ in range(n_det)]

    rollout_scores = np.zeros(len(candidates), dtype=np.float64)
    vleaf_scores = np.zeros(len(candidates), dtype=np.float64)

    # ── Rollout path ──────────────────────────────────────────────────────
    for det in dets:
        policy = random.choice(rollout_mix)
        cls = ROLLOUT_FACTORIES[policy]
        agents = [cls(env.level_rank) for _ in range(4)]
        for i, move in enumerate(candidates):
            sim = _clone_env(det)
            sim.step(move)
            rollout_scores[i] += _rollout_limited(sim, agents, depth_limit, player)

    # ── V-leaf path (batched) ─────────────────────────────────────────────
    leaf_states, leaf_idx = [], []
    for det in dets:
        for i, move in enumerate(candidates):
            sim = _clone_env(det)
            sim.step(move)
            if sim.done:
                vleaf_scores[i] += float(sim.get_rewards()[player])
            else:
                leaf_states.append(_encode_for_value(sim, player))
                leaf_idx.append(i)
    if leaf_states:
        device = value_net.v_net[0].weight.device
        st = torch.from_numpy(np.stack(leaf_states)).to(device)
        with torch.no_grad():
            v = value_net.value(st).cpu().numpy()
        for k, i in enumerate(leaf_idx):
            vleaf_scores[i] += float(v[k])

    return rollout_scores, vleaf_scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--decisions", type=int, default=200)
    ap.add_argument("--n-det", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=3)
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
        use_search=False,  # we'll call _policy_top_k manually
        top_k=args.top_k,
        device=torch.device("cpu"),
    )
    value_net = oracle.net

    env = GuanDanEnv(level_rank=Rank.TWO)
    spearmans, kls, argmax_agree, max_pi_agreement = [], [], [], []
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

            r_scores, v_scores = score_both(
                env, p, cands, value_net, args.n_det, rollout_mix, rng,
            )
            pi_r = _softmax(r_scores)
            pi_v = _softmax(v_scores)

            spearmans.append(_spearman(r_scores, v_scores))
            kls.append(_kl(pi_r, pi_v))
            argmax_agree.append(int(np.argmax(r_scores) == np.argmax(v_scores)))
            max_pi_agreement.append(float(min(pi_r.max(), pi_v.max())))

            rec += 1
            if rec % 20 == 0:
                print(f"  {rec}/{args.decisions}  "
                      f"r={np.mean(spearmans):.3f}  "
                      f"argmax_agree={np.mean(argmax_agree):.1%}  "
                      f"KL={np.mean(kls):.3f}")

            # Play whichever method's argmax (use rollout for stability)
            env.step(cands[int(np.argmax(r_scores))])

    sp = np.array(spearmans)
    print()
    print("=" * 60)
    print(f"Decisions compared:    {len(sp)}")
    print(f"Spearman r:            mean={sp.mean():.3f}  median={np.median(sp):.3f}  "
          f"std={sp.std():.3f}")
    print(f"  r >= 0.9:            {(sp >= 0.9).mean():.1%}")
    print(f"  r >= 0.5:            {(sp >= 0.5).mean():.1%}")
    print(f"  r <= 0:              {(sp <= 0.0).mean():.1%}")
    print(f"Argmax agreement:      {np.mean(argmax_agree):.1%}")
    print(f"Mean KL(rollout||vleaf): {np.mean(kls):.3f}")
    print("=" * 60)
    if sp.mean() > 0.85 and np.mean(argmax_agree) > 0.80:
        print("VERDICT: V-leaf is a faithful proxy for rollouts. Use V-leaf.")
    elif sp.mean() > 0.5:
        print("VERDICT: Signals partially agree. Consider mixed approach.")
    else:
        print("VERDICT: Signals diverge. V-leaf is NOT a faithful proxy.")


if __name__ == "__main__":
    main()
