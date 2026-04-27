"""Stage 0: bot-agnostic distillation into the pvguan architecture.

Collects supervisor decisions via partner-visible self-play, trains the
ActorCriticNet actor head on masked-softmax cross-entropy, then validates
against acceptance gates before saving the checkpoint.

Usage:
    PYTHONPATH=ml/src python ml/scripts/train/distill_pvguan.py \\
        --supervisor oracle --supervisor-search on \\
        --n-decisions 500000 \\
        --output ml/checkpoints/pvguan_distilled_oracle.pt

    # smoke test (~5 min on M1)
    PYTHONPATH=ml/src python ml/scripts/train/distill_pvguan.py \\
        --supervisor jidan --supervisor-search off \\
        --n-decisions 5000 --epochs 2 --workers 2 \\
        --output ml/checkpoints/pvguan_distilled_smoke.pt
"""

from __future__ import annotations

import argparse
import hashlib
import math
import multiprocessing as mp
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from guandan.agents import JidanBot
from guandan.agents.base import Agent
from guandan.agents.greedy_bot import GreedyBot
from guandan.agents.heuristic_bot import HeuristicBot
from guandan.agents.partner_oracle_bot import _reflect_env, PartnerOracleBot
from guandan.agents.strategic_bot import StrategicBot
from guandan.cards import Rank
from guandan.combos import Combo
from guandan.game import GuanDanEnv
from guandan.pvguan.actor_critic import ActorCriticNet, get_device
from guandan.pvguan.encoders import (
    ACTION_DIM,
    ACTOR_DIM,
    encode_action,
    encode_actor_pair_features,
)

MAX_LEGAL_PER_SAMPLE = 64


# ─── Partner-Visible Info State ───────────────────────────────────────────────

@dataclass(frozen=True)
class PartnerVisibleInfoState:
    """Supervisor input: own+partner hands + public info. No opponent hands."""
    own_hand:       frozenset
    partner_hand:   frozenset
    public_history: tuple          # (actor, combo) tuples
    hand_counts:    tuple          # 4 ints
    is_out:         tuple          # 4 bools
    finish_order:   tuple          # seats in finish order
    level_rank:     int
    current_trick:  Any            # Combo | None
    legal_moves:    tuple          # Combo objects

    @classmethod
    def from_env(cls, env: GuanDanEnv, player: int) -> "PartnerVisibleInfoState":
        partner = (player + 2) % 4
        return cls(
            own_hand=frozenset(env.hands[player]),
            partner_hand=frozenset(env.hands[partner]),
            public_history=tuple(env.move_history[-30:]),
            hand_counts=tuple(len(env.hands[p]) for p in range(4)),
            is_out=tuple(env.is_out),
            finish_order=tuple(env.finish_order),
            level_rank=env.level_rank,
            current_trick=env.current_trick,
            legal_moves=tuple(env.legal_moves(player)),
        )


# ─── Supervisor registry ──────────────────────────────────────────────────────

def _build_supervisor(name: str, use_search: bool, n_det: int, checkpoint: str | None) -> Agent:
    lr = Rank.TWO
    if name == "oracle":
        ckpt = checkpoint or "ml/checkpoints/jidan_policy.pt"
        return PartnerOracleBot(ckpt, level_rank=lr, use_search=use_search, n_det=n_det)
    if name == "jidan":
        return JidanBot(level_rank=lr)
    if name == "greedy":
        return GreedyBot()
    if name == "heuristic":
        return HeuristicBot()
    if name == "strategic":
        return StrategicBot()
    try:
        from guandan.agents.yaoji_bot import YaojiBot
        if name == "yaoji":
            return YaojiBot(level_rank=lr)
    except ImportError:
        pass
    raise ValueError(f"Unknown supervisor: {name!r}")


def _combo_key(combo: Combo) -> tuple:
    return (int(combo.type), tuple(sorted((c.rank, c.suit, c.deck) for c in combo.cards)))


# ─── Data collection worker ───────────────────────────────────────────────────

def _worker(args: tuple) -> list[dict]:
    n_dec, level_range, seed, sup_name, use_search, n_det, ckpt_path = args
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

    from guandan.agents import JidanBot
    from guandan.agents.partner_oracle_bot import PartnerOracleBot, _reflect_env
    from guandan.cards import Rank
    from guandan.combos import Combo
    from guandan.game import GuanDanEnv
    from guandan.pvguan.encoders import encode_action, encode_actor_pair_features

    random.seed(seed)
    np.random.seed(seed)

    supervisor = _build_supervisor(sup_name, use_search, n_det, ckpt_path)
    env = GuanDanEnv(level_rank=Rank.TWO)
    samples: list[dict] = []

    while len(samples) < n_dec:
        level_rank = random.randint(*level_range)
        env.level_rank = level_rank
        env.reset()

        while not env.done:
            player = env.current_player
            # Always collect from both canonical teams; reflect if needed
            needs_reflect = player in (1, 3)
            canonical_player = player ^ 1 if needs_reflect else player
            enc_env = _reflect_env(env) if needs_reflect else env

            legal = enc_env.legal_moves(canonical_player)
            if len(legal) <= 1:
                pick = supervisor.act(enc_env, canonical_player)
                env.step(pick)
                continue

            # Get supervisor pick from partner-visible info state
            info = PartnerVisibleInfoState.from_env(enc_env, canonical_player)
            sup_pick = supervisor.act(enc_env, canonical_player)
            pick_key = _combo_key(sup_pick)

            pick_idx_full = next(
                (i for i, a in enumerate(legal) if _combo_key(a) == pick_key), 0
            )

            # Cap legal moves to bound memory
            if len(legal) > MAX_LEGAL_PER_SAMPLE:
                scored = sorted(range(len(legal)),
                                key=lambda i: sum(c.rank for c in legal[i].cards))
                keep_idx = scored[:MAX_LEGAL_PER_SAMPLE]
                if pick_idx_full not in keep_idx:
                    keep_idx[-1] = pick_idx_full
                keep_idx = sorted(keep_idx)
                capped_legal = [legal[i] for i in keep_idx]
                pick_idx = keep_idx.index(pick_idx_full)
            else:
                capped_legal = legal
                pick_idx = pick_idx_full

            hand = list(enc_env.hands[canonical_player])
            sa = np.stack([
                encode_actor_pair_features(enc_env, canonical_player, a, capped_legal)
                for a in capped_legal
            ]).astype(np.float32)
            ac = np.stack([
                encode_action(a, hand, enc_env.level_rank)
                for a in capped_legal
            ]).astype(np.float32)

            samples.append({"states": sa, "actions": ac, "pick_idx": pick_idx})
            env.step(sup_pick)

    return samples[:n_dec]


def collect_data(
    n_decisions: int,
    workers: int,
    sup_name: str,
    use_search: bool,
    n_det: int,
    ckpt_path: str | None,
    seed: int,
    level_range: tuple[int, int] = (2, 14),
) -> list[dict]:
    per_worker = math.ceil(n_decisions / workers)
    print(f"  data: {workers} workers × {per_worker} decisions (target {n_decisions})",
          flush=True)
    t0 = time.time()

    arg_list = [
        (per_worker, level_range, seed + w, sup_name, use_search, n_det, ckpt_path)
        for w in range(workers)
    ]

    if workers == 1:
        all_samples = _worker(arg_list[0])
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            results = pool.map(_worker, arg_list)
        all_samples = [s for chunk in results for s in chunk]

    elapsed = time.time() - t0
    print(f"  collected {len(all_samples)} samples in {elapsed:.1f}s "
          f"({len(all_samples)/elapsed:.0f}/s)", flush=True)
    return all_samples[:n_decisions]


# ─── Training step ────────────────────────────────────────────────────────────

def _train_step(net: ActorCriticNet, batch: list[dict], device, opt=None) -> tuple[float, int, int]:
    B = len(batch)
    max_K = max(s["states"].shape[0] for s in batch)

    sa = torch.zeros(B, max_K, ACTOR_DIM, dtype=torch.float32, device=device)
    ac = torch.zeros(B, max_K, ACTION_DIM, dtype=torch.float32, device=device)
    mask = torch.zeros(B, max_K, dtype=torch.bool, device=device)
    targets = torch.zeros(B, dtype=torch.long, device=device)

    for i, s in enumerate(batch):
        K = s["states"].shape[0]
        sa[i, :K]    = torch.from_numpy(s["states"])
        ac[i, :K]    = torch.from_numpy(s["actions"])
        mask[i, :K]  = True
        targets[i]   = s["pick_idx"]

    sa_flat = sa.view(B * max_K, ACTOR_DIM)
    ac_flat = ac.view(B * max_K, ACTION_DIM)
    logits_flat = net.actor_head(torch.cat([sa_flat, ac_flat], dim=-1))
    logits = logits_flat.view(B, max_K).masked_fill(~mask, -1e9)

    loss = F.cross_entropy(logits, targets)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    pred = logits.argmax(dim=-1)
    correct = (pred == targets).sum().item()
    return loss.item(), correct, B


# ─── Main training loop ───────────────────────────────────────────────────────

def train_distillation(
    samples: list[dict],
    epochs: int,
    batch_size: int,
    lr: float,
    hidden: int,
    device: torch.device,
    label_smoothing: float = 0.1,
    val_split: float = 0.10,
    patience: int = 3,
    seed: int = 0,
) -> tuple[ActorCriticNet, dict]:
    net = ActorCriticNet(hidden=hidden).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)

    n_params = sum(p.numel() for p in net.parameters())
    print(f"  net: {n_params:,} params on {device} (hidden={hidden})")

    rng = random.Random(seed)
    n = len(samples)
    idxs = list(range(n))
    rng.shuffle(idxs)
    n_val = int(n * val_split)
    val_idxs = idxs[:n_val]
    train_idxs = idxs[n_val:]
    print(f"  split: {len(train_idxs)} train / {len(val_idxs)} val")

    best_val_acc = 0.0
    best_state = None
    stale = 0

    for epoch in range(epochs):
        rng.shuffle(train_idxs)
        net.train()
        tr_loss = tr_correct = tr_total = 0
        for bs in range(0, len(train_idxs), batch_size):
            batch = [samples[i] for i in train_idxs[bs: bs + batch_size]]
            l, c, t = _train_step(net, batch, device, opt=opt)
            tr_loss += l; tr_correct += c; tr_total += t
        sched.step()

        net.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for bs in range(0, len(val_idxs), batch_size):
                batch = [samples[i] for i in val_idxs[bs: bs + batch_size]]
                _, c, t = _train_step(net, batch, device, opt=None)
                val_correct += c; val_total += t

        tr_acc = tr_correct / max(tr_total, 1)
        val_acc = val_correct / max(val_total, 1)
        print(f"  epoch {epoch+1}/{epochs}  tr_acc={tr_acc:.3f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stop at epoch {epoch+1}")
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    # Final val accuracy
    net.eval()
    val_correct = val_total = 0
    with torch.no_grad():
        for bs in range(0, len(val_idxs), batch_size):
            batch = [samples[i] for i in val_idxs[bs: bs + batch_size]]
            _, c, t = _train_step(net, batch, device, opt=None)
            val_correct += c; val_total += t
    final_val_acc = val_correct / max(val_total, 1)

    return net, {"val_acc": final_val_acc, "best_val_acc": best_val_acc}


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("distill_pvguan")
    parser.add_argument("--supervisor", default="oracle",
        choices=["oracle", "jidan", "greedy", "heuristic", "strategic", "yaoji"])
    parser.add_argument("--supervisor-search", default="on", choices=["on", "off"])
    parser.add_argument("--supervisor-pimc-n-det", type=int, default=16)
    parser.add_argument("--supervisor-checkpoint", default=None,
        help="Path to supervisor's model checkpoint (for oracle)")
    parser.add_argument("--n-decisions", type=int, default=500_000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label-mode", default="hard", choices=["hard", "soft", "hybrid"])
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--output", default="ml/checkpoints/pvguan_distilled.pt")
    args = parser.parse_args()

    use_search = args.supervisor_search == "on"
    device = get_device()
    print(f"[distill_pvguan] supervisor={args.supervisor} search={use_search} "
          f"n_det={args.supervisor_pimc_n_det} device={device}")

    # 1. Collect labels
    samples = collect_data(
        n_decisions=args.n_decisions,
        workers=args.workers,
        sup_name=args.supervisor,
        use_search=use_search,
        n_det=args.supervisor_pimc_n_det,
        ckpt_path=args.supervisor_checkpoint,
        seed=args.seed,
    )

    # 2. Train actor head
    net, metrics = train_distillation(
        samples=samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        device=device,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
    )

    val_acc = metrics["val_acc"]
    print(f"\n[distill_pvguan] val_acc={val_acc:.4f}")

    # 3. Acceptance gates
    if val_acc < 0.90:
        print(f"WARN: val_acc {val_acc:.3f} < 0.90 (target ≥ 0.95 for production)")
    if val_acc < 0.95:
        print(f"WARN: argmax agreement {val_acc:.3f} below 95% gate — review before PPO")

    # 4. Save checkpoint
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "actor_dim":   ACTOR_DIM,
        "action_dim":  ACTION_DIM,
        "critic_dim":  875,
        "hidden":      args.hidden,
        "supervisor":  args.supervisor,
        "supervisor_search": use_search,
        "supervisor_pimc_n_det": args.supervisor_pimc_n_det,
        "n_samples":   len(samples),
        "val_acc":     val_acc,
        "label_mode":  args.label_mode,
        "label_smoothing": args.label_smoothing,
    }

    torch.save({
        "actor_state_dict": net.actor_head.state_dict(),
        "full_state_dict":  net.state_dict(),
        "metadata": metadata,
    }, output_path)
    print(f"[distill_pvguan] saved → {output_path}")
    print(f"  metadata: {metadata}")


if __name__ == "__main__":
    main()
