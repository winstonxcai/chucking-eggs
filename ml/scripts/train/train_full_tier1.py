#!/usr/bin/env python
"""Full 477-dim retraining for Tier 1 partner visibility.

Trains QNetworkLSTM(d_state=477) with ALL parameters unfrozen, warm-started
from the expanded production checkpoint. Very low LR + warmup prevent the
catastrophic forgetting that killed v1-v3.

Key differences from v1-v3:
  - LR 3e-6 (10x lower than v1's 3e-5)
  - 2000-episode linear warmup from 0
  - Self-play: all 4 seats use 477-dim encoding
  - Aux loss 0.1 weight (same as base training)
  - Eval + save every 1000 episodes

Usage:
    PYTHONPATH=ml/src python ml/scripts/train/train_full_tier1.py \\
        --base-checkpoint ml/checkpoints/prod_03_29_11_51.pt \\
        --run-name full_tier1_v1
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

log = logging.getLogger(__name__)


def _episode_worker(
    lead_sd: dict,
    follow_sd: dict,
    lstm_hidden: int,
    mlp_hidden: int,
    epsilon: float,
    opponent_name: str | None,
    d_state: int,
) -> list:
    """Run one episode in a subprocess (CPU only, no MPS)."""
    from guandan.agents import make_agent
    from guandan.game import GuanDanEnv
    from guandan.training.encoding import encode_action, encode_history, encode_opponent_cards
    from guandan.training.q_network import QNetworkLSTM
    from guandan.training.visibility.encoding import encode_state_tier1

    device = torch.device("cpu")
    q_lead = QNetworkLSTM(d_state=d_state, lstm_hidden=lstm_hidden, hidden=mlp_hidden).to(device)
    q_follow = QNetworkLSTM(d_state=d_state, lstm_hidden=lstm_hidden, hidden=mlp_hidden).to(device)
    q_lead.load_state_dict(lead_sd)
    q_follow.load_state_dict(follow_sd)
    q_lead.eval(); q_follow.eval()

    env = GuanDanEnv()
    opp = make_agent(opponent_name, env.level_rank) if opponent_name else None
    return play_episode(env, q_lead, q_follow, epsilon, device,
                        encode_state_tier1, encode_action, encode_history,
                        encode_opponent_cards, env.level_rank, opponent=opp)


def _setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(run_dir / "train.log", mode="a")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    if not any(isinstance(h, logging.StreamHandler) and h.stream is sys.stderr
               for h in root.handlers):
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        root.addHandler(sh)


def _fmt(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def play_episode(env, q_lead, q_follow, epsilon, device,
                 encode_fn, encode_action_fn, encode_history_fn,
                 encode_opp_fn, level_rank, opponent=None):
    """Play one episode.

    If opponent is None: self-play, all 4 seats use 477-dim encoding and
    contribute transitions. WR stays ~50% by symmetry — good for warming up
    the partner features but not for measuring improvement.

    If opponent is provided: seats {0,2} use our agent (477-dim), seats {1,3}
    use the opponent. Only seats {0,2} contribute transitions. WR reflects
    real improvement from partner visibility.
    """
    env.reset()
    our_seats = (0, 1, 2, 3) if opponent is None else (0, 2)
    transitions: dict[int, list] = {p: [] for p in our_seats}

    while not env.done:
        player = env.current_player

        if opponent is not None and player in (1, 3):
            env.step(opponent.act(env, player))
            continue

        legal = env.legal_moves()
        is_leading = env.current_trick is None
        q_net = q_lead if is_leading else q_follow

        state_enc = encode_fn(env, player)
        hand = env.hands[player]
        action_encs = np.array([encode_action_fn(m, hand, level_rank) for m in legal])
        history, hist_len = encode_history_fn(env, player, level_rank)

        if random.random() < epsilon:
            idx = random.randint(0, len(legal) - 1)
        else:
            with torch.no_grad():
                B = len(legal)
                s = torch.tensor(state_enc, device=device).unsqueeze(0).expand(B, -1)
                a = torch.tensor(action_encs, device=device)
                h = torch.tensor(history, device=device).unsqueeze(0).expand(B, -1, -1)
                hl = torch.tensor([hist_len], dtype=torch.long, device=device).expand(B)
                idx = q_net(s, a, h, hl).argmax().item()

        opp_cards = encode_opp_fn(env, player)
        transitions[player].append((state_enc, action_encs[idx], history, hist_len, opp_cards))
        env.step(legal[idx])

    rewards = env.get_rewards()
    all_trans = []
    for player, tlist in transitions.items():
        r = rewards[player]
        for s, a, h, hl, oc in tlist:
            all_trans.append((s, a, h, hl, r, oc))
    return all_trans


_PARTNER_COL_START = 60
_PARTNER_COL_END = 120  # 60 + PARTNER_HAND_DIM


def _mask_nonpartner_grads(model) -> None:
    """Zero all gradients except partner columns (60:120) in first input layers.

    Used in Stage 1 to ensure only the 60 new partner input weights update.
    All pretrained parameters stay frozen.
    """
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if name in ("mlp.0.weight", "hand_pred.0.weight"):
            g = param.grad
            g[:, :_PARTNER_COL_START] = 0
            g[:, _PARTNER_COL_END:] = 0
        else:
            param.grad.zero_()


def train_step(q_net, buf, optimizer, batch_size, device,
               aux_weight=0.1, q_weight=1.0, partner_only=False):
    if len(buf) < batch_size:
        return None
    batch = buf.sample(batch_size, device)
    hist_emb = q_net.encode_history(batch["history"], batch["hist_len"])

    total = torch.tensor(0.0, device=device)
    q_loss_val = 0.0

    if q_weight > 0:
        q_pred = q_net.forward_from_embedding(batch["state"], batch["action"], hist_emb)
        q_loss = torch.nn.functional.mse_loss(q_pred, batch["return"])
        total = total + q_weight * q_loss
        q_loss_val = q_loss.item()

    hand_pred = q_net.predict_opponent_cards(batch["state"], hist_emb)
    aux_loss = torch.nn.functional.binary_cross_entropy(hand_pred, batch["opponent_cards"])
    total = total + aux_weight * aux_loss

    optimizer.zero_grad()
    total.backward()
    if partner_only:
        _mask_nonpartner_grads(q_net)
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
    optimizer.step()
    return q_loss_val, aux_loss.item()


def evaluate(q_lead, q_follow, device, n_games, opponent_name,
             encode_fn, encode_action_fn, encode_history_fn, level_rank):
    from guandan.agents import make_agent
    from guandan.game import GuanDanEnv
    env = GuanDanEnv()
    opp = make_agent(opponent_name, env.level_rank)
    wins = finish_12 = finish_13 = finish_14 = 0

    for _ in tqdm(range(n_games), desc=f"eval vs {opponent_name}", unit="game",
                  leave=False, file=sys.stderr, dynamic_ncols=True):
        env.reset()
        while not env.done:
            player = env.current_player
            if player in (1, 3):
                env.step(opp.act(env, player))
                continue
            legal = env.legal_moves()
            is_leading = env.current_trick is None
            q_net = q_lead if is_leading else q_follow
            state_enc = encode_fn(env, player)
            hand = env.hands[player]
            action_encs = np.array([encode_action_fn(m, hand, level_rank) for m in legal])
            history, hist_len = encode_history_fn(env, player, level_rank)
            with torch.no_grad():
                B = len(legal)
                s = torch.tensor(state_enc, device=device).unsqueeze(0).expand(B, -1)
                a = torch.tensor(action_encs, device=device)
                h = torch.tensor(history, device=device).unsqueeze(0).expand(B, -1, -1)
                hl = torch.tensor([hist_len], dtype=torch.long, device=device).expand(B)
                idx = q_net(s, a, h, hl).argmax().item()
            env.step(legal[idx])

        rewards = env.get_rewards()
        if rewards[0] + rewards[2] > 0:
            wins += 1
        fo = env.finish_order
        tp = sorted(fo.index(p) for p in (0, 2))
        key = (tp[0] + 1, tp[1] + 1)
        if key == (1, 2): finish_12 += 1
        elif key == (1, 3): finish_13 += 1
        elif key == (1, 4): finish_14 += 1

    return {"winrate": wins / n_games, "finish_12": finish_12,
            "finish_13": finish_13, "finish_14": finish_14}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--run-name", default="full_tier1_v1")
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--lr-warmup", type=int, default=2000,
                        help="Linear LR warmup over this many episodes")
    parser.add_argument("--lr-end", type=float, default=3e-7)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--buffer-capacity", type=int, default=300000)
    parser.add_argument("--prefill", type=int, default=500)
    parser.add_argument("--train-steps", type=int, default=4)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--epsilon-start", type=float, default=0.08)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-games", type=int, default=300)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--lstm-hidden", type=int, default=256)
    parser.add_argument("--mlp-hidden", type=int, default=1024)
    parser.add_argument("--opponent", type=str, default=None,
                        help="Opponent agent for seats {1,3}. None = self-play.")
    parser.add_argument("--stage1-episodes", type=int, default=5000,
                        help="Episodes for Stage 1: aux-only, partner columns only.")
    parser.add_argument("--stage1-lr", type=float, default=1e-4,
                        help="LR for Stage 1 (partner column aux training).")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel episode workers (CPU). 1 = sequential.")
    args = parser.parse_args()

    from guandan.game import GuanDanEnv
    from guandan.training.encoding import encode_action, encode_history, encode_opponent_cards
    from guandan.training.q_network import QNetworkLSTM, get_device
    from guandan.training.replay import ReplayBuffer
    from guandan.training.visibility.encoding import STATE_DIM_TIER1, encode_state_tier1
    from guandan.training.visibility.expand import expand_checkpoint

    device = get_device()
    run_dir = Path("ml/runs") / args.run_name
    _setup_logging(run_dir)

    log.info("=" * 60)
    log.info("Full Tier 1 Retraining — 477-dim, two-stage")
    log.info("=" * 60)
    log.info("Device: %s | d_state: %d", device, STATE_DIM_TIER1)
    log.info("Stage 1: %d ep, aux-only, partner cols only, lr=%.1e",
             args.stage1_episodes, args.stage1_lr)
    log.info("Stage 2: %d ep, Q+aux, all params, lr=%.1e (warmup %d) → %.1e",
             args.episodes, args.lr, args.lr_warmup, args.lr_end)
    log.info("Workers: %d", args.n_workers)

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Expand checkpoint
    expanded_path = run_dir / "checkpoint_expanded.pt"
    if expanded_path.exists():
        log.info("Using existing expanded checkpoint: %s", expanded_path)
        ckpt = torch.load(expanded_path, map_location="cpu", weights_only=True)
    else:
        log.info("Expanding %s → %s", args.base_checkpoint, expanded_path)
        ckpt = expand_checkpoint(args.base_checkpoint, expanded_path)

    q_lead = QNetworkLSTM(d_state=STATE_DIM_TIER1,
                          lstm_hidden=args.lstm_hidden,
                          hidden=args.mlp_hidden).to(device)
    q_follow = QNetworkLSTM(d_state=STATE_DIM_TIER1,
                            lstm_hidden=args.lstm_hidden,
                            hidden=args.mlp_hidden).to(device)
    q_lead.load_state_dict(ckpt["lead"])
    q_follow.load_state_dict(ckpt["follow"])
    log.info("Loaded expanded weights. All parameters trainable.")
    log.info("Total params per net: %d", sum(p.numel() for p in q_lead.parameters()))

    # Verify starting WR
    q_lead.eval(); q_follow.eval()
    env = GuanDanEnv()
    level_rank = env.level_rank
    result = evaluate(q_lead, q_follow, device, 500, "jidan",
                      encode_state_tier1, encode_action, encode_history, level_rank)
    log.info("Starting WR vs jidan: %.1f%% (1-2: %d, 1-3: %d, 1-4: %d)",
             result["winrate"] * 100, result["finish_12"],
             result["finish_13"], result["finish_14"])

    from guandan.agents import make_agent
    opp_agent = make_agent(args.opponent, env.level_rank) if args.opponent else None
    log.info("Opponent: %s", args.opponent if args.opponent else "self-play")

    buffer = ReplayBuffer(capacity=args.buffer_capacity, d_state=STATE_DIM_TIER1)
    opt_lead = torch.optim.Adam(q_lead.parameters(), lr=args.lr)
    opt_follow = torch.optim.Adam(q_follow.parameters(), lr=args.lr)

    use_parallel = args.n_workers > 1
    pool = ProcessPoolExecutor(max_workers=args.n_workers) if use_parallel else None

    def _collect_batch(epsilon: float, n: int) -> list:
        """Collect n episodes, returns flat list of transitions."""
        if use_parallel:
            lead_sd = {k: v.cpu() for k, v in q_lead.state_dict().items()}
            follow_sd = {k: v.cpu() for k, v in q_follow.state_dict().items()}
            futs = [pool.submit(_episode_worker, lead_sd, follow_sd,
                                args.lstm_hidden, args.mlp_hidden, epsilon,
                                args.opponent, STATE_DIM_TIER1)
                    for _ in range(n)]
            return [t for fut in futs for t in fut.result()]
        else:
            all_trans = []
            for _ in range(n):
                all_trans.extend(play_episode(
                    env, q_lead, q_follow, epsilon, device,
                    encode_state_tier1, encode_action, encode_history,
                    encode_opponent_cards, level_rank, opponent=opp_agent))
            return all_trans

    # Prefill
    log.info("Prefilling buffer with %d episodes (%d workers)...",
             args.prefill, args.n_workers)
    t_pre = time.time()
    trans = _collect_batch(args.epsilon_start, args.prefill)
    for t in trans:
        buffer.push(*t)
    log.info("Prefill done: %d transitions (%.1f/ep) in %s",
             len(buffer), len(trans) / args.prefill, _fmt(time.time() - t_pre))

    # ── Stage 1: aux-only, partner columns only ──────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("Stage 1: aux-only, partner columns, lr=%.1e", args.stage1_lr)
    log.info("=" * 60)
    opt_s1_lead = torch.optim.Adam(q_lead.parameters(), lr=args.stage1_lr)
    opt_s1_follow = torch.optim.Adam(q_follow.parameters(), lr=args.stage1_lr)
    t_start = time.time()
    loss_q_sum = loss_aux_sum = 0.0
    loss_count = 0

    step = args.n_workers if use_parallel else 1
    pbar1 = tqdm(total=args.stage1_episodes, desc="stage1", file=sys.stderr)
    ep = 0
    while ep < args.stage1_episodes:
        q_lead.eval(); q_follow.eval()
        trans = _collect_batch(args.epsilon_start, step)
        for t in trans:
            buffer.push(*t)
        q_lead.train(); q_follow.train()
        for _ in range(args.train_steps * step):
            r = train_step(q_lead, buffer, opt_s1_lead, args.batch_size, device,
                           aux_weight=1.0, q_weight=0.0, partner_only=True)
            train_step(q_follow, buffer, opt_s1_follow, args.batch_size, device,
                       aux_weight=1.0, q_weight=0.0, partner_only=True)
            if r is not None:
                loss_aux_sum += r[1]; loss_count += 1
        ep += step
        pbar1.update(step)
        if ep % 200 < step and loss_count > 0:
            elapsed = time.time() - t_start
            log.info("S1 Ep %5d | aux=%.4f | buf=%d | %.1f ep/s | %s",
                     ep, loss_aux_sum / loss_count, len(buffer),
                     ep / elapsed, _fmt(elapsed))
            loss_aux_sum = loss_count = 0
        if ep % args.eval_interval < step:
            q_lead.eval(); q_follow.eval()
            res = evaluate(q_lead, q_follow, device, args.eval_games, "jidan",
                           encode_state_tier1, encode_action, encode_history, level_rank)
            log.info("S1 EVAL Ep %5d | vs_jidan=%.1f%% | 1-2: %d/%d (%.0f%%)",
                     ep, res["winrate"] * 100, res["finish_12"], args.eval_games,
                     res["finish_12"] / args.eval_games * 100)
    pbar1.close()
    log.info("Stage 1 complete.")

    # ── Stage 2: full Q+aux, all params, very low LR ─────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("Stage 2: Q+aux, all params, lr=%.1e → %.1e", args.lr, args.lr_end)
    log.info("=" * 60)
    opt_lead = torch.optim.Adam(q_lead.parameters(), lr=args.lr)
    opt_follow = torch.optim.Adam(q_follow.parameters(), lr=args.lr)
    t_start = time.time()
    best_wr = result["winrate"]
    loss_q_sum = loss_aux_sum = 0.0
    loss_count = 0
    metrics = []

    pbar = tqdm(total=args.episodes, desc="stage2", file=sys.stderr)
    ep = 0
    while ep < args.episodes:
        # LR schedule: linear warmup then cosine decay
        if ep <= args.lr_warmup:
            lr = args.lr * max(ep, 1) / args.lr_warmup
        else:
            frac = (ep - args.lr_warmup) / max(1, args.episodes - args.lr_warmup)
            lr = args.lr_end + 0.5 * (args.lr - args.lr_end) * (1 + np.cos(np.pi * frac))
        for opt in (opt_lead, opt_follow):
            for pg in opt.param_groups:
                pg["lr"] = lr

        frac_eps = min(1.0, ep / (0.8 * args.episodes))
        epsilon = args.epsilon_start + (args.epsilon_end - args.epsilon_start) * frac_eps

        # Collect
        q_lead.eval(); q_follow.eval()
        trans = _collect_batch(epsilon, step)
        for t in trans:
            buffer.push(*t)

        # Train
        q_lead.train(); q_follow.train()
        for _ in range(args.train_steps * step):
            r_lead = train_step(q_lead, buffer, opt_lead, args.batch_size, device,
                                aux_weight=args.aux_weight, q_weight=1.0, partner_only=False)
            train_step(q_follow, buffer, opt_follow, args.batch_size, device,
                       aux_weight=args.aux_weight, q_weight=1.0, partner_only=False)
            if r_lead is not None:
                loss_q_sum += r_lead[0]
                loss_aux_sum += r_lead[1]
                loss_count += 1

        ep += step
        pbar.update(step)

        if loss_count > 0:
            elapsed = time.time() - t_start
            pbar.set_postfix_str(
                f"q={loss_q_sum/loss_count:.3f} aux={loss_aux_sum/loss_count:.3f} "
                f"lr={lr:.1e} {ep/elapsed:.1f}ep/s"
            )

        # Periodic log
        if ep % 200 < step and loss_count > 0:
            elapsed = time.time() - t_start
            log.info("Ep %6d | q=%.4f aux=%.4f | lr=%.2e e=%.3f | buf=%d | %.1f ep/s | %s",
                     ep, loss_q_sum / loss_count, loss_aux_sum / loss_count,
                     lr, epsilon, len(buffer), ep / elapsed, _fmt(elapsed))
            loss_q_sum = loss_aux_sum = 0.0
            loss_count = 0

        # Eval
        if ep % args.eval_interval < step:
            q_lead.eval(); q_follow.eval()
            elapsed = time.time() - t_start
            t_eval = time.time()
            res = evaluate(q_lead, q_follow, device, args.eval_games, "jidan",
                           encode_state_tier1, encode_action, encode_history, level_rank)
            wr = res["winrate"]
            log.info("-" * 60)
            log.info("EVAL Ep %6d | vs_jidan=%.1f%% | 1-2: %d/%d (%.0f%%) | "
                     "elapsed %s | eval took %s",
                     ep, wr * 100, res["finish_12"], args.eval_games,
                     res["finish_12"] / args.eval_games * 100,
                     _fmt(elapsed), _fmt(time.time() - t_eval))
            log.info("     jidan detail: 1-2=%d 1-3=%d 1-4=%d",
                     res["finish_12"], res["finish_13"], res["finish_14"])
            log.info("-" * 60)

            entry = {"episode": ep, "elapsed_s": elapsed,
                     "wr_jidan": wr, **{f"finish_{k}_jidan": v
                                        for k, v in res.items() if "finish" in k}}
            metrics.append(entry)
            with open(run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")

            if wr > best_wr:
                best_wr = wr
                best_path = run_dir / "checkpoint_best.pt"
                torch.save({"lead": q_lead.state_dict(), "follow": q_follow.state_dict(),
                            "episode": ep, "wr_jidan": wr, "d_state": STATE_DIM_TIER1},
                           best_path)
                log.info("  *** New best: %.1f%% → %s ***", wr * 100, best_path)

        # Save checkpoint
        if ep % args.save_interval < step:
            path = run_dir / f"checkpoint_ep{ep}.pt"
            torch.save({"lead": q_lead.state_dict(), "follow": q_follow.state_dict(),
                        "episode": ep, "d_state": STATE_DIM_TIER1}, path)
            log.info("Saved: %s", path)

    pbar.close()
    if pool is not None:
        pool.shutdown(wait=False)
    total = time.time() - t_start
    final_path = run_dir / "model_final.pt"
    torch.save({"lead": q_lead.state_dict(), "follow": q_follow.state_dict(),
                "episode": args.episodes, "d_state": STATE_DIM_TIER1,
                "best_wr_jidan": best_wr}, final_path)
    log.info("=" * 60)
    log.info("Done. Best vs jidan: %.1f%%. Wall time: %s", best_wr * 100, _fmt(total))
    log.info("Final: %s", final_path)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
