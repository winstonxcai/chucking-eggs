"""Generate AZ self-play training data using PartnerOracleBot (Direction D, Phase 2).

Each decision records:
  base_state   [480]       encode_state_tier1_team (no action-specific flags) — for V head
  cand_states  [K, 489]    per-candidate state (base + 9-dim behavior flags) — for Q-policy
  actions      [K, 160]    top-K policy candidates, encoded
  pi_search    [K]         softmax of PIMC scores over those K candidates
  n_cands      scalar      actual K (rest of actions/pi_search rows are zero-padded)
  neg_states   [K_neg,489] per-negative state (for ranking loss)
  neg_actions  [K_neg,160] hard-negative non-candidate actions
  n_neg        scalar      actual count of negatives
  z            scalar      final reward for the acting player

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
from guandan.training.visibility import (
    STATE_DIM_TIER1_TEAM,
    STATE_DIM_TIER1_TEAM_WITH_FLAGS,
)

TOP_K_MAX = 10  # maximum top_k supported (for padding)
K_NEG = 5       # hard negative non-candidates to store (for ranking loss)


# ── Worker ─────────────────────────────────────────────────────────────────────

def _worker(args: tuple) -> list[dict]:
    """Run games in one worker process, return list of decision dicts."""
    checkpoint_path, n_decisions, n_det, top_k, seed, use_value_leaf, policy_checkpoint, pi_temp = args

    import random
    import numpy as np_w
    import torch

    random.seed(seed)
    np_w.random.seed(seed)
    torch.manual_seed(seed)

    from guandan.agents.partner_oracle_bot import PartnerOracleBot, _reflect_env
    from guandan.cards import Rank
    from guandan.game import GuanDanEnv
    from guandan.training import ACTION_DIM, QValueNet, encode_action
    from guandan.training.visibility import (
        STATE_DIM_TIER1_TEAM,
        encode_state_tier1_team,
        encode_state_tier1_team_with_flags,
    )

    oracle = PartnerOracleBot(
        checkpoint_path=checkpoint_path,
        level_rank=Rank.TWO,
        use_search=True,
        n_det=n_det,
        top_k=top_k,
        use_belief=True,   # match eval-time belief (H1-H4 constraints improve PIMC realism)
        n_workers=1,  # single-process PIMC — we're already inside a worker
        use_value_leaf=use_value_leaf,
        device=torch.device("cpu"),  # avoid GPU contention across self-play workers
    )

    if policy_checkpoint is not None:
        # Hybrid oracle: use a separate (clean) policy net for top-K candidate
        # selection while the value head from checkpoint_path handles V-at-leaf.
        # This decouples policy filtering from value estimation, preventing
        # Q-drift in the value checkpoint from contaminating candidate quality.
        pol_ckpt = torch.load(policy_checkpoint, map_location="cpu", weights_only=True)
        pol_cfg = pol_ckpt["config"]
        pol_net = QValueNet(
            d_state=pol_cfg["d_state"],
            d_action=pol_cfg["d_action"],
            hidden=pol_cfg["hidden"],
        )
        pol_net.load_state_dict(pol_ckpt["state_dict"], strict=False)
        pol_net.eval()
        # oracle._search.value_net still references the original net (gen-1 V head)
        # because Python bound the reference at PartnerPIMCBot.__init__ time.
        # Replacing oracle.net only affects _policy_top_k / _score_actions.
        oracle.net = pol_net

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

            # Softmax → pi_search (with optional temperature for softer targets)
            scores_arr = scores_arr / pi_temp
            scores_arr -= scores_arr.max()
            exp_s = np_w.exp(scores_arr)
            pi_search = (exp_s / exp_s.sum()).astype(np_w.float32)

            move = candidates[int(np_w.argmax(scores_arr))]

            # Encode state (seat-reflected for {1,3})
            if player in (0, 2):
                enc_env, enc_player = env, player
            else:
                enc_env, enc_player = _reflect_env(env), player ^ 1

            base_state = encode_state_tier1_team(enc_env, enc_player).astype(np_w.float32)

            # Per-candidate state with action-specific behavior flags
            cand_states = np_w.stack([
                encode_state_tier1_team_with_flags(enc_env, enc_player, a, legal)
                for a in candidates
            ]).astype(np_w.float32)
            actions = np_w.stack([
                encode_action(a, enc_env.hands[enc_player], env.level_rank)
                for a in candidates
            ]).astype(np_w.float32)

            # Sample non-candidate legal moves as hard negatives for ranking loss.
            cand_ids = {id(c) for c in candidates}
            non_cands = [a for a in legal if id(a) not in cand_ids]
            n_neg = min(len(non_cands), K_NEG)
            if n_neg > 0:
                neg_sample = random.sample(non_cands, n_neg)
                neg_states = np_w.stack([
                    encode_state_tier1_team_with_flags(enc_env, enc_player, a, legal)
                    for a in neg_sample
                ]).astype(np_w.float32)
                neg_actions = np_w.stack([
                    encode_action(a, enc_env.hands[enc_player], env.level_rank)
                    for a in neg_sample
                ]).astype(np_w.float32)
            else:
                neg_states = np_w.zeros((0, cand_states.shape[1]), dtype=np_w.float32)
                neg_actions = np_w.zeros((0, actions.shape[1]), dtype=np_w.float32)

            game_buf.append({
                "base_state": base_state,        # [480]   — for V head
                "cand_states": cand_states,      # [K, 489] — for Q-policy
                "actions": actions,              # [K, 160]
                "pi_search": pi_search,          # [K]
                "n_cands": len(candidates),
                "neg_states": neg_states,        # [n_neg, 489]
                "neg_actions": neg_actions,      # [n_neg, 160]
                "n_neg": n_neg,
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
    use_value_leaf: bool = False,
    policy_checkpoint: str | None = None,
    pi_temp: float = 1.0,
) -> dict[str, np.ndarray]:
    per_worker = math.ceil(n_decisions / workers)
    args_list = [
        (checkpoint, per_worker, n_det, top_k, seed + w, use_value_leaf, policy_checkpoint, pi_temp)
        for w in range(workers)
    ]
    print(
        f"  self-play: {workers} workers × {per_worker} decisions each "
        f"(target {n_decisions}, n_det={n_det}, K={top_k}, pi_temp={pi_temp})",
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

    # Pack into numpy arrays (zero-pad to top_k / K_NEG)
    N = len(all_decisions)
    K = top_k
    D_FLAGS = STATE_DIM_TIER1_TEAM_WITH_FLAGS
    base_states = np.zeros((N, STATE_DIM_TIER1_TEAM), dtype=np.float32)
    cand_states = np.zeros((N, K, D_FLAGS), dtype=np.float32)
    actions = np.zeros((N, K, ACTION_DIM), dtype=np.float32)
    pi_search = np.zeros((N, K), dtype=np.float32)
    n_cands = np.zeros(N, dtype=np.int32)
    neg_states = np.zeros((N, K_NEG, D_FLAGS), dtype=np.float32)
    neg_actions = np.zeros((N, K_NEG, ACTION_DIM), dtype=np.float32)
    n_neg = np.zeros(N, dtype=np.int32)
    z = np.zeros(N, dtype=np.float32)

    for i, d in enumerate(all_decisions):
        base_states[i] = d["base_state"]
        k = min(d["n_cands"], K)
        cand_states[i, :k] = d["cand_states"][:k]
        actions[i, :k] = d["actions"][:k]
        pi_search[i, :k] = d["pi_search"][:k]
        n_cands[i] = k
        kn = min(d.get("n_neg", 0), K_NEG)
        if kn > 0:
            neg_states[i, :kn] = d["neg_states"][:kn]
            neg_actions[i, :kn] = d["neg_actions"][:kn]
        n_neg[i] = kn
        z[i] = d["z"]

    return {
        "base_states": base_states,
        "cand_states": cand_states,
        "actions": actions,
        "pi_search": pi_search,
        "n_cands": n_cands,
        "neg_states": neg_states,
        "neg_actions": neg_actions,
        "n_neg": n_neg,
        "z": z,
        "config": np.array([D_FLAGS, ACTION_DIM, K], dtype=np.int32),
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
    p.add_argument("--use-value-leaf", action="store_true",
                   help="Use V(s) at leaf instead of Jidan rollouts (AZ gen-2+).")
    p.add_argument("--policy-checkpoint", default=None,
                   help="Separate policy checkpoint for top-K candidate selection "
                        "(hybrid oracle: clean policy filter + value-leaf from --checkpoint).")
    p.add_argument("--pi-temp", type=float, default=1.0,
                   help="Temperature for PIMC score softmax (>1 = softer π_search targets). "
                        "Use 2.0-3.0 with V-at-leaf to prevent training target collapse.")
    args = p.parse_args()

    print(f"AZ self-play data generation")
    print(f"  checkpoint: {args.checkpoint}")
    if args.policy_checkpoint:
        print(f"  policy-checkpoint: {args.policy_checkpoint}  (hybrid oracle)")
    print(f"  decisions={args.decisions}  n_det={args.n_det}  K={args.top_k}  "
          f"workers={args.workers}", flush=True)

    data = generate(
        checkpoint=args.checkpoint,
        n_decisions=args.decisions,
        n_det=args.n_det,
        top_k=args.top_k,
        workers=args.workers,
        seed=args.seed,
        use_value_leaf=args.use_value_leaf,
        policy_checkpoint=args.policy_checkpoint,
        pi_temp=args.pi_temp,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    N = data['base_states'].shape[0]
    print(f"\nSaved {N} decisions → {out}")
    print(f"  base_states:  {data['base_states'].shape}")
    print(f"  cand_states:  {data['cand_states'].shape}")
    print(f"  actions:      {data['actions'].shape}")
    print(f"  pi_search:    {data['pi_search'].shape}")
    print(f"  z:  mean={data['z'].mean():.3f}  std={data['z'].std():.3f}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
