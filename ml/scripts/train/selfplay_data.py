"""Generate AZ self-play training data using PartnerOracleBot (Direction D, Phase 2).

Each decision records:
  state     [480]     encode_state_tier1_team from acting player's perspective
  actions   [K, 160]  top-K policy candidates, encoded
  pi_search [K]       softmax of PIMC scores over those K candidates
  n_cands   scalar    actual K (rest of actions/pi_search rows are zero-padded)
  z         scalar    final reward for the acting player

All 4 seats use the same PartnerOracleBot (one network, seat reflection for {1,3}).
Only decisions where PIMC search ran (|candidates| > 1) are recorded.

Usage:
    PYTHONPATH=ml/src python ml/scripts/train/selfplay_data.py \\
        --checkpoint ml/checkpoints/jidan_policy.pt \\
        --decisions 30000 --out ml/data/selfplay_gen1.npz \\
        --n-det 30 --top-k 3 --workers 8

    # smoke test (~100 decisions)
    PYTHONPATH=ml/src python ml/scripts/train/selfplay_data.py \\
        --checkpoint ml/checkpoints/jidan_policy.pt \\
        --decisions 500 --out /tmp/selfplay_smoke.npz \\
        --n-det 5 --top-k 3 --workers 2
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from guandan.training import ACTION_DIM
from guandan.training.visibility import STATE_DIM_TIER1_TEAM

TOP_K_MAX = 10  # maximum top_k supported (for padding)


# ── Worker ─────────────────────────────────────────────────────────────────────

def _worker(args: tuple) -> list[dict]:
    """Run games in one worker process, return list of decision dicts."""
    checkpoint_path, n_decisions, n_det, top_k, seed = args

    import random
    import numpy as np_w
    import torch

    random.seed(seed)
    np_w.random.seed(seed)
    torch.manual_seed(seed)

    from guandan.agents.partner_oracle_bot import PartnerOracleBot, _reflect_env
    from guandan.cards import Rank
    from guandan.game import GuanDanEnv
    from guandan.training import encode_action
    from guandan.training.visibility import encode_state_tier1_team

    oracle = PartnerOracleBot(
        checkpoint_path=checkpoint_path,
        level_rank=Rank.TWO,
        use_search=True,
        n_det=n_det,
        top_k=top_k,
        use_belief=False,
        n_workers=1,  # single-process PIMC — we're already inside a worker
    )

    env = GuanDanEnv(level_rank=Rank.TWO)
    all_decisions: list[dict] = []

    while len(all_decisions) < n_decisions:
        env.reset()
        game_buf: list[dict] = []  # decisions this game (z backfilled at end)

        while not env.done:
            player = env.current_player
            legal = env.legal_moves(player)

            if len(legal) == 1:
                env.step(legal[0])
                continue

            # Policy top-K candidates
            candidates = oracle._policy_top_k(env, player, legal)

            if oracle._search is None or len(candidates) <= 1:
                # No search or single candidate — play policy argmax, skip recording
                import numpy as _np
                q = oracle._score_actions(env, player, candidates if len(candidates) > 1 else legal)
                move = (candidates if len(candidates) > 1 else legal)[int(_np.argmax(q))]
                env.step(move)
                continue

            # PIMC search: get per-candidate scores
            scores = oracle._search._score_candidates(env, player, candidates)
            scores_arr = np_w.array(scores, dtype=np_w.float32)

            # Softmax → pi_search
            scores_arr -= scores_arr.max()
            exp_s = np_w.exp(scores_arr)
            pi_search = (exp_s / exp_s.sum()).astype(np_w.float32)

            move = candidates[int(np_w.argmax(scores_arr))]

            # Encode state (seat-reflected for {1,3})
            if player in (0, 2):
                enc_env, enc_player = env, player
            else:
                enc_env, enc_player = _reflect_env(env), player ^ 1

            state = encode_state_tier1_team(enc_env, enc_player).astype(np_w.float32)

            actions = np_w.stack([
                encode_action(a, enc_env.hands[enc_player], env.level_rank)
                for a in candidates
            ]).astype(np_w.float32)

            game_buf.append({
                "state": state,          # [480]
                "actions": actions,      # [K, 160]
                "pi_search": pi_search,  # [K]
                "n_cands": len(candidates),
                "player": player,
            })

            env.step(move)

        # Backfill z from final rewards
        rewards = env.get_rewards()
        for d in game_buf:
            d["z"] = float(rewards[d["player"]])
            del d["player"]

        all_decisions.extend(game_buf)

    return all_decisions[:n_decisions]


# ── Main ───────────────────────────────────────────────────────────────────────

def generate(
    checkpoint: str,
    n_decisions: int,
    n_det: int,
    top_k: int,
    workers: int,
    seed: int,
) -> dict[str, np.ndarray]:
    per_worker = math.ceil(n_decisions / workers)
    args_list = [
        (checkpoint, per_worker, n_det, top_k, seed + w)
        for w in range(workers)
    ]
    print(
        f"  self-play: {workers} workers × {per_worker} decisions each "
        f"(target {n_decisions}, n_det={n_det}, K={top_k})",
        flush=True,
    )

    t0 = time.time()
    if workers == 1:
        all_decisions = _worker(args_list[0])
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            chunks = pool.map(_worker, args_list)
        all_decisions = [d for chunk in chunks for d in chunk]

    all_decisions = all_decisions[:n_decisions]
    elapsed = time.time() - t0
    print(
        f"  self-play: {len(all_decisions)} decisions in {elapsed:.1f}s "
        f"({len(all_decisions)/elapsed:.1f} dec/s)",
        flush=True,
    )

    # Pack into numpy arrays (zero-pad to top_k)
    N = len(all_decisions)
    K = top_k
    states = np.zeros((N, STATE_DIM_TIER1_TEAM), dtype=np.float32)
    actions = np.zeros((N, K, ACTION_DIM), dtype=np.float32)
    pi_search = np.zeros((N, K), dtype=np.float32)
    n_cands = np.zeros(N, dtype=np.int32)
    z = np.zeros(N, dtype=np.float32)

    for i, d in enumerate(all_decisions):
        states[i] = d["state"]
        k = min(d["n_cands"], K)
        actions[i, :k] = d["actions"][:k]
        pi_search[i, :k] = d["pi_search"][:k]
        n_cands[i] = k
        z[i] = d["z"]

    return {
        "states": states,
        "actions": actions,
        "pi_search": pi_search,
        "n_cands": n_cands,
        "z": z,
        "config": np.array([STATE_DIM_TIER1_TEAM, ACTION_DIM, K], dtype=np.int32),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="AZ self-play data generation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--decisions", type=int, default=30000)
    p.add_argument("--out", required=True)
    p.add_argument("--n-det", type=int, default=30)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"AZ self-play data generation")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  decisions={args.decisions}  n_det={args.n_det}  K={args.top_k}  "
          f"workers={args.workers}", flush=True)

    data = generate(
        checkpoint=args.checkpoint,
        n_decisions=args.decisions,
        n_det=args.n_det,
        top_k=args.top_k,
        workers=args.workers,
        seed=args.seed,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    print(f"\nSaved {data['states'].shape[0]} decisions → {out}")
    print(f"  states:    {data['states'].shape}")
    print(f"  actions:   {data['actions'].shape}")
    print(f"  pi_search: {data['pi_search'].shape}")
    print(f"  z:  mean={data['z'].mean():.3f}  std={data['z'].std():.3f}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
