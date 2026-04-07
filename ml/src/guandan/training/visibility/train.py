"""Tier 1 training: fine-tune with partner hand visibility.

Uses the same play_episode / train_step / evaluate pattern as the base trainer,
but with encode_state_tier1 (477 dims) instead of encode_state (417 dims).
No behavior flags — matching the base checkpoint architecture.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

log = logging.getLogger(__name__)

from ...agents import make_agent
from ...game import GuanDanEnv
from ..encoding import (
    ACTION_DIM,
    encode_action,
    encode_history,
    encode_opponent_cards,
)
from ..q_network import QNetworkLSTM, get_device
from ..replay import ReplayBuffer
from ..train import train_step
from .encoding import STATE_DIM_TIER1, encode_state_tier1


def play_episode_tier1(
    env: GuanDanEnv,
    q_lead: QNetworkLSTM,
    q_follow: QNetworkLSTM,
    epsilon: float,
    device: torch.device,
    opponent=None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int, float, np.ndarray]]:
    """Play one game with partner-visible encoding.

    If opponent is None: symmetric self-play — all 4 seats use Tier 1 encoding.
      Returns transitions from all 4 seats. Keeps the return distribution
      similar to base model's self-play, preventing catastrophic forgetting.

    If opponent is provided: seats {0,2} use Tier 1, seats {1,3} use opponent.
      Returns transitions from seats {0,2} only.
    """
    env.reset()
    selfplay = opponent is None
    seats = range(4) if selfplay else (0, 2)
    transitions: dict[int, list[tuple]] = {p: [] for p in seats}

    while not env.done:
        player = env.current_player

        if not selfplay and player in (1, 3):
            env.step(opponent.act(env, player))
            continue

        legal = env.legal_moves()
        is_leading = env.current_trick is None

        state_enc = encode_state_tier1(env, player)  # [477]
        hand = env.hands[player]
        action_encs = np.array(
            [encode_action(m, hand, env.level_rank) for m in legal]
        )
        history, hist_len = encode_history(env, player, env.level_rank)

        q_net = q_lead if is_leading else q_follow

        if random.random() < epsilon:
            idx = random.randint(0, len(legal) - 1)
        else:
            with torch.no_grad():
                B = len(legal)
                s = (
                    torch.tensor(state_enc, device=device)
                    .unsqueeze(0)
                    .expand(B, -1)
                )
                a = torch.tensor(action_encs, device=device)
                h = (
                    torch.tensor(history, device=device)
                    .unsqueeze(0)
                    .expand(B, -1, -1)
                )
                hl = torch.tensor(
                    [hist_len], dtype=torch.long, device=device
                ).expand(B)
                idx = q_net(s, a, h, hl).argmax().item()

        opp_cards = encode_opponent_cards(env, player)
        transitions[player].append(
            (state_enc, action_encs[idx], history, hist_len, opp_cards)
        )
        env.step(legal[idx])

    rewards = env.get_rewards()
    all_trans = []
    for player, tlist in transitions.items():
        r = rewards[player]
        for s, a, h, hl, oc in tlist:
            all_trans.append((s, a, h, hl, r, oc))
    return all_trans


def evaluate_tier1(
    q_lead: QNetworkLSTM,
    q_follow: QNetworkLSTM,
    device: torch.device,
    n_games: int = 500,
    opponent: str = "jidan",
) -> dict:
    """Evaluate Tier 1 agent (seats {0,2}) vs opponent (seats {1,3})."""
    env = GuanDanEnv()
    opp_agent = make_agent(opponent, env.level_rank)

    wins = 0
    total_reward = 0.0
    finish_12 = finish_13 = finish_14 = 0

    for _ in tqdm(range(n_games), desc=f"eval vs {opponent}", unit="game",
                  leave=False, file=sys.stderr, dynamic_ncols=True):
        env.reset()
        while not env.done:
            player = env.current_player
            if player in (1, 3):
                env.step(opp_agent.act(env, player))
                continue

            legal = env.legal_moves()
            is_leading = env.current_trick is None
            q_net = q_lead if is_leading else q_follow

            state_enc = encode_state_tier1(env, player)
            hand = env.hands[player]
            action_encs = np.array(
                [encode_action(m, hand, env.level_rank) for m in legal]
            )
            history, hist_len = encode_history(env, player, env.level_rank)

            with torch.no_grad():
                B = len(legal)
                s = (
                    torch.tensor(state_enc, device=device)
                    .unsqueeze(0)
                    .expand(B, -1)
                )
                a = torch.tensor(action_encs, device=device)
                h = (
                    torch.tensor(history, device=device)
                    .unsqueeze(0)
                    .expand(B, -1, -1)
                )
                hl = torch.tensor(
                    [hist_len], dtype=torch.long, device=device
                ).expand(B)
                idx = q_net(s, a, h, hl).argmax().item()

            env.step(legal[idx])

        rewards = env.get_rewards()
        team_reward = rewards[0] + rewards[2]
        if team_reward > 0:
            wins += 1
        total_reward += team_reward

        fo = env.finish_order
        team_pos = sorted(fo.index(p) for p in (0, 2))
        key = (team_pos[0] + 1, team_pos[1] + 1)
        if key == (1, 2):
            finish_12 += 1
        elif key == (1, 3):
            finish_13 += 1
        elif key == (1, 4):
            finish_14 += 1

    return {
        "winrate": wins / n_games,
        "avg_reward": total_reward / n_games,
        "finish_12": finish_12,
        "finish_13": finish_13,
        "finish_14": finish_14,
        "n_games": n_games,
    }


def _append_metrics(run_dir: Path, entry: dict) -> None:
    with open(run_dir / "metrics.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def _setup_logging(run_dir: Path) -> None:
    """Configure logging to both stderr and run_dir/train.log."""
    run_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    # File handler — persistent log
    fh = logging.FileHandler(run_dir / "train.log", mode="a")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Stream handler (stderr) — only if none exists yet
    if not any(isinstance(h, logging.StreamHandler) and h.stream is sys.stderr
               for h in root.handlers):
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        root.addHandler(sh)


def _fmt_elapsed(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def run_training(
    q_lead: QNetworkLSTM,
    q_follow: QNetworkLSTM,
    config: dict,
    run_dir: Path,
    device: torch.device,
) -> None:
    """Main Tier 1 training loop: prefill → stage 1 (strategic) → stage 2 (jidan)."""
    _setup_logging(run_dir)

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    log.info("=" * 60)
    log.info("Tier 1 Training — Partner Visibility")
    log.info("=" * 60)
    log.info("Device: %s", device)
    log.info("Config: %s", json.dumps(config, indent=2))
    log.info("Run dir: %s", run_dir)
    log.info("STATE_DIM_TIER1 = %d", STATE_DIM_TIER1)

    buffer = ReplayBuffer(
        capacity=config["buffer_capacity"],
        d_state=STATE_DIM_TIER1,
    )

    # --- Frozen backbone: only train mlp.0 + hand_pred.0 initially ---
    freeze_episodes = config.get("freeze_backbone_episodes", 0)
    if freeze_episodes > 0:
        log.info("Frozen backbone for first %d episodes (only mlp.0 + hand_pred.0 trainable)", freeze_episodes)

    def _set_backbone_frozen(model: QNetworkLSTM, frozen: bool) -> None:
        """Freeze/unfreeze everything except mlp.0 and hand_pred.0."""
        for name, param in model.named_parameters():
            if name.startswith("mlp.0.") or name.startswith("hand_pred.0."):
                param.requires_grad = True  # always trainable
            else:
                param.requires_grad = not frozen

    def _count_trainable(model: QNetworkLSTM) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    backbone_frozen = freeze_episodes > 0
    if backbone_frozen:
        _set_backbone_frozen(q_lead, True)
        _set_backbone_frozen(q_follow, True)
        log.info("Trainable params (frozen): %d / %d",
                 _count_trainable(q_lead),
                 sum(p.numel() for p in q_lead.parameters()))

    # Only pass trainable params to optimizer — rebuilt when unfreezing
    opt_lead = torch.optim.Adam(
        [p for p in q_lead.parameters() if p.requires_grad], lr=config["lr_start"]
    )
    opt_follow = torch.optim.Adam(
        [p for p in q_follow.parameters() if p.requires_grad], lr=config["lr_start"]
    )

    env = GuanDanEnv()
    t_start = time.time()

    use_selfplay = config.get("selfplay", False)
    log.info("Training mode: %s", "self-play (symmetric)" if use_selfplay else "vs frozen opponents")

    # --- Pre-fill buffer ---
    prefill_eps = config["prefill_episodes"]
    log.info("Pre-filling buffer with %d %s episodes...",
             prefill_eps, "self-play" if use_selfplay else "on-policy")
    t_prefill = time.time()
    prefill_opp = None if use_selfplay else make_agent(config["stage1_opponent"], env.level_rank)
    q_lead.eval()
    q_follow.eval()
    prefill_trans = 0
    for _ in tqdm(range(prefill_eps), desc="prefill", file=sys.stderr):
        trans = play_episode_tier1(
            env, q_lead, q_follow,
            epsilon=config["epsilon_start"], device=device,
            opponent=prefill_opp,
        )
        prefill_trans += len(trans)
        for s, a, h, hl, r, oc in trans:
            buffer.push(s, a, h, hl, r, oc)
    prefill_secs = time.time() - t_prefill
    log.info(
        "Buffer pre-filled: %d transitions from %d episodes "
        "(%.1f trans/ep, %.1f ep/s, %s)",
        len(buffer), prefill_eps,
        prefill_trans / prefill_eps,
        prefill_eps / prefill_secs,
        _fmt_elapsed(prefill_secs),
    )

    # --- Stage loop ---
    stages = [
        (config["stage1_opponent"], config["stage1_episodes"], config["stage1_gate"]),
        (config["stage2_opponent"], config["stage2_episodes"], None),
    ]

    total_ep = 0
    best_wr_jidan = 0.0
    # Running loss accumulators (reset each log interval)
    loss_q_sum = 0.0
    loss_aux_sum = 0.0
    loss_count = 0
    log_interval = 200  # log loss summary every N episodes

    for stage_idx, (opp_name, stage_episodes, gate) in enumerate(stages, 1):
        log.info("")
        log.info("=" * 60)
        log.info("Stage %d: vs %s (%d episodes, gate=%s)",
                 stage_idx, opp_name, stage_episodes,
                 f"{gate:.0%}" if gate else "none")
        log.info("=" * 60)
        opp = None if use_selfplay else make_agent(opp_name, env.level_rank)
        t_stage = time.time()

        eps_start = config["epsilon_start"]
        eps_end = config["epsilon_end"]
        total_all = config["stage1_episodes"] + config["stage2_episodes"]
        lr_start = config["lr_start"]
        lr_end = config["lr_end"]
        warmup_eps = config.get("lr_warmup_episodes", 0)
        warmup_lr_start = config.get("lr_warmup_start", lr_end)

        if warmup_eps > 0:
            log.info("LR warmup: %s → %s over %d episodes",
                     f"{warmup_lr_start:.1e}", f"{lr_start:.1e}", warmup_eps)

        pbar = tqdm(range(stage_episodes), desc=f"stage{stage_idx}", file=sys.stderr)
        for ep_in_stage in pbar:
            total_ep += 1
            frac = min(1.0, total_ep / (0.85 * total_all))
            epsilon = eps_start + (eps_end - eps_start) * frac

            # LR schedule: linear warmup → cosine decay
            if total_ep <= warmup_eps:
                # Linear warmup from warmup_lr_start to lr_start
                warmup_frac = total_ep / warmup_eps
                lr = warmup_lr_start + (lr_start - warmup_lr_start) * warmup_frac
            else:
                # Cosine decay from lr_start to lr_end
                decay_ep = total_ep - warmup_eps
                decay_total = total_all - warmup_eps
                lr = lr_end + 0.5 * (lr_start - lr_end) * (
                    1 + np.cos(np.pi * decay_ep / decay_total)
                )
            for opt in (opt_lead, opt_follow):
                for pg in opt.param_groups:
                    pg["lr"] = lr

            # Unfreeze backbone after freeze_episodes
            if backbone_frozen and total_ep > freeze_episodes:
                backbone_frozen = False
                _set_backbone_frozen(q_lead, False)
                _set_backbone_frozen(q_follow, False)
                # Rebuild optimizers with all params
                opt_lead = torch.optim.Adam(q_lead.parameters(), lr=lr)
                opt_follow = torch.optim.Adam(q_follow.parameters(), lr=lr)
                log.info(
                    "*** Backbone unfrozen at ep %d. Trainable params: %d ***",
                    total_ep, _count_trainable(q_lead),
                )

            # Collect episode
            q_lead.eval()
            q_follow.eval()
            trans = play_episode_tier1(
                env, q_lead, q_follow, epsilon, device, opponent=opp
            )
            for s, a, h, hl, r, oc in trans:
                buffer.push(s, a, h, hl, r, oc)

            # Train steps — capture losses
            q_lead.train()
            q_follow.train()
            bs = config["batch_size"]
            for _ in range(config["train_steps_per_episode"]):
                res_lead = train_step(
                    q_lead, buffer, opt_lead, bs, device, config["aux_weight"]
                )
                train_step(
                    q_follow, buffer, opt_follow, bs, device, config["aux_weight"]
                )
                if res_lead is not None:
                    loss_q_sum += res_lead[0]
                    loss_aux_sum += res_lead[1]
                    loss_count += 1

            # Update tqdm postfix with running stats
            if loss_count > 0:
                elapsed = time.time() - t_start
                eps_per_sec = total_ep / elapsed
                pbar.set_postfix_str(
                    f"q={loss_q_sum/loss_count:.4f} "
                    f"aux={loss_aux_sum/loss_count:.4f} "
                    f"e={epsilon:.3f} "
                    f"{eps_per_sec:.1f}ep/s"
                )

            # Periodic loss log
            if total_ep % log_interval == 0 and loss_count > 0:
                avg_q = loss_q_sum / loss_count
                avg_aux = loss_aux_sum / loss_count
                elapsed = time.time() - t_start
                log.info(
                    "Ep %6d | q_loss=%.4f aux_loss=%.4f | "
                    "e=%.3f lr=%.2e | buf=%d | "
                    "%.1f ep/s | %s elapsed",
                    total_ep, avg_q, avg_aux,
                    epsilon, lr, len(buffer),
                    total_ep / elapsed, _fmt_elapsed(elapsed),
                )
                loss_q_sum = 0.0
                loss_aux_sum = 0.0
                loss_count = 0

            # Eval
            eval_interval = config["eval_interval"]
            if (ep_in_stage + 1) % eval_interval == 0 or ep_in_stage == stage_episodes - 1:
                q_lead.eval()
                q_follow.eval()
                t_eval = time.time()

                result_stage = evaluate_tier1(
                    q_lead, q_follow, device,
                    n_games=config["eval_games"], opponent=opp_name,
                )
                result_jidan = evaluate_tier1(
                    q_lead, q_follow, device,
                    n_games=config["eval_games"], opponent="jidan",
                )
                eval_secs = time.time() - t_eval

                # Partner column L2 norm (lead + follow)
                partner_norm_lead = torch.norm(
                    q_lead.mlp[0].weight[:, 60:120]
                ).item()
                partner_norm_follow = torch.norm(
                    q_follow.mlp[0].weight[:, 60:120]
                ).item()

                wr_stage = result_stage["winrate"]
                wr_jidan = result_jidan["winrate"]
                elapsed = time.time() - t_start
                remaining_ep = total_all - total_ep
                eta_secs = remaining_ep / (total_ep / elapsed) if total_ep > 0 else 0

                log.info("-" * 60)
                log.info(
                    "EVAL Ep %6d | vs_%s=%.1f%% | vs_jidan=%.1f%% | "
                    "1-2: %d/%d (%.0f%%) | "
                    "partner_L2: lead=%.4f follow=%.4f",
                    total_ep, opp_name, wr_stage * 100, wr_jidan * 100,
                    result_jidan["finish_12"], result_jidan["n_games"],
                    result_jidan["finish_12"] / result_jidan["n_games"] * 100,
                    partner_norm_lead, partner_norm_follow,
                )
                log.info(
                    "     stage_detail: 1-2=%d 1-3=%d 1-4=%d | "
                    "jidan_detail: 1-2=%d 1-3=%d 1-4=%d | "
                    "eval took %s | elapsed %s | ETA %s",
                    result_stage["finish_12"], result_stage["finish_13"],
                    result_stage["finish_14"],
                    result_jidan["finish_12"], result_jidan["finish_13"],
                    result_jidan["finish_14"],
                    _fmt_elapsed(eval_secs), _fmt_elapsed(elapsed),
                    _fmt_elapsed(eta_secs),
                )
                log.info("-" * 60)

                _append_metrics(run_dir, {
                    "episode": total_ep,
                    "stage": stage_idx,
                    "timestamp": time.time(),
                    "elapsed_s": elapsed,
                    f"wr_{opp_name}": wr_stage,
                    "wr_jidan": wr_jidan,
                    "avg_reward_jidan": result_jidan["avg_reward"],
                    "finish_12_jidan": result_jidan["finish_12"],
                    "finish_13_jidan": result_jidan["finish_13"],
                    "finish_14_jidan": result_jidan["finish_14"],
                    f"finish_12_{opp_name}": result_stage["finish_12"],
                    "partner_column_l2_lead": partner_norm_lead,
                    "partner_column_l2_follow": partner_norm_follow,
                    "epsilon": epsilon,
                    "lr": lr,
                    "buffer_size": len(buffer),
                })

                if wr_jidan > best_wr_jidan:
                    best_wr_jidan = wr_jidan
                    best_path = run_dir / "checkpoint_best.pt"
                    torch.save({
                        "lead": q_lead.state_dict(),
                        "follow": q_follow.state_dict(),
                        "episode": total_ep,
                        "wr_jidan": wr_jidan,
                        "d_state": STATE_DIM_TIER1,
                    }, best_path)
                    log.info(
                        "  *** New best: %.1f%% vs jidan → %s ***",
                        wr_jidan * 100, best_path,
                    )

                # Stage gate check
                if gate is not None and wr_stage >= gate:
                    log.info(
                        "  Stage gate reached: %.1f%% >= %.0f%%. Advancing.",
                        wr_stage * 100, gate * 100,
                    )
                    pbar.close()
                    break

            # Periodic save
            if total_ep % config["save_interval"] == 0:
                path = run_dir / f"checkpoint_ep{total_ep}.pt"
                torch.save({
                    "lead": q_lead.state_dict(),
                    "follow": q_follow.state_dict(),
                    "episode": total_ep,
                    "d_state": STATE_DIM_TIER1,
                }, path)
                log.info("Saved checkpoint: %s", path)

        else:
            pbar.close()

        stage_secs = time.time() - t_stage
        log.info(
            "Stage %d complete: %d episodes in %s (%.1f ep/s)",
            stage_idx, ep_in_stage + 1, _fmt_elapsed(stage_secs),
            (ep_in_stage + 1) / stage_secs,
        )

    # Final save
    total_secs = time.time() - t_start
    final_path = run_dir / "model_final.pt"
    torch.save({
        "lead": q_lead.state_dict(),
        "follow": q_follow.state_dict(),
        "episode": total_ep,
        "d_state": STATE_DIM_TIER1,
        "best_wr_jidan": best_wr_jidan,
    }, final_path)
    log.info("")
    log.info("=" * 60)
    log.info("Training complete")
    log.info("  Total episodes: %d", total_ep)
    log.info("  Best vs jidan: %.1f%%", best_wr_jidan * 100)
    log.info("  Wall time: %s", _fmt_elapsed(total_secs))
    log.info("  Final checkpoint: %s", final_path)
    log.info("=" * 60)
