"""Tier 1 training: fine-tune with partner hand visibility.

Uses the same play_episode / train_step / evaluate pattern as the base trainer,
but with encode_state_tier1 (477 dims) instead of encode_state (417 dims).
No behavior flags — matching the base checkpoint architecture.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

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
    opponent,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int, float, np.ndarray]]:
    """Play one game with partner-visible encoding on seats {0,2}.

    Opponent plays seats {1,3} via opponent.act().
    Returns (state, action, history, hist_len, reward, opp_cards) tuples
    for RL team only.
    """
    env.reset()
    transitions: dict[int, list[tuple]] = {0: [], 2: []}

    while not env.done:
        player = env.current_player

        if player in (1, 3):
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


def run_training(
    q_lead: QNetworkLSTM,
    q_follow: QNetworkLSTM,
    config: dict,
    run_dir: Path,
    device: torch.device,
) -> None:
    """Main Tier 1 training loop: prefill → stage 1 (strategic) → stage 2 (jidan)."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    buffer = ReplayBuffer(
        capacity=config["buffer_capacity"],
        d_state=STATE_DIM_TIER1,
    )

    opt_lead = torch.optim.Adam(q_lead.parameters(), lr=config["lr_start"])
    opt_follow = torch.optim.Adam(q_follow.parameters(), lr=config["lr_start"])

    env = GuanDanEnv()

    # --- Pre-fill buffer ---
    prefill_eps = config["prefill_episodes"]
    print(f"Pre-filling buffer with {prefill_eps} on-policy episodes...")
    prefill_opp = make_agent(config["stage1_opponent"], env.level_rank)
    q_lead.eval()
    q_follow.eval()
    for _ in tqdm(range(prefill_eps), desc="prefill", file=sys.stderr):
        trans = play_episode_tier1(
            env, q_lead, q_follow,
            epsilon=config["epsilon_start"], device=device,
            opponent=prefill_opp,
        )
        for s, a, h, hl, r, oc in trans:
            buffer.push(s, a, h, hl, r, oc)
    print(f"Buffer pre-filled: {len(buffer):,} transitions")

    # --- Stage loop ---
    stages = [
        (config["stage1_opponent"], config["stage1_episodes"], config["stage1_gate"]),
        (config["stage2_opponent"], config["stage2_episodes"], None),
    ]

    total_ep = 0
    best_wr_jidan = 0.0

    for stage_idx, (opp_name, stage_episodes, gate) in enumerate(stages, 1):
        print(f"\n=== Stage {stage_idx}: vs {opp_name} ({stage_episodes} episodes) ===")
        opp = make_agent(opp_name, env.level_rank)

        eps_start = config["epsilon_start"]
        eps_end = config["epsilon_end"]
        total_all = config["stage1_episodes"] + config["stage2_episodes"]
        lr_start = config["lr_start"]
        lr_end = config["lr_end"]

        pbar = tqdm(range(stage_episodes), desc=f"stage{stage_idx}", file=sys.stderr)
        for ep_in_stage in pbar:
            total_ep += 1
            frac = min(1.0, total_ep / (0.85 * total_all))
            epsilon = eps_start + (eps_end - eps_start) * frac

            # Cosine LR decay
            lr = lr_end + 0.5 * (lr_start - lr_end) * (
                1 + np.cos(np.pi * total_ep / total_all)
            )
            for opt in (opt_lead, opt_follow):
                for pg in opt.param_groups:
                    pg["lr"] = lr

            # Collect episode
            q_lead.eval()
            q_follow.eval()
            trans = play_episode_tier1(
                env, q_lead, q_follow, epsilon, device, opponent=opp
            )
            for s, a, h, hl, r, oc in trans:
                buffer.push(s, a, h, hl, r, oc)

            # Train steps
            q_lead.train()
            q_follow.train()
            bs = config["batch_size"]
            for _ in range(config["train_steps_per_episode"]):
                train_step(q_lead, buffer, opt_lead, bs, device, config["aux_weight"])
                train_step(q_follow, buffer, opt_follow, bs, device, config["aux_weight"])

            # Eval
            eval_interval = config["eval_interval"]
            if (ep_in_stage + 1) % eval_interval == 0 or ep_in_stage == stage_episodes - 1:
                q_lead.eval()
                q_follow.eval()

                result_stage = evaluate_tier1(
                    q_lead, q_follow, device,
                    n_games=config["eval_games"], opponent=opp_name,
                )
                result_jidan = evaluate_tier1(
                    q_lead, q_follow, device,
                    n_games=config["eval_games"], opponent="jidan",
                )

                # Partner column L2 norm
                partner_norm = torch.norm(
                    q_lead.mlp[0].weight[:, 60:120]
                ).item()

                wr_stage = result_stage["winrate"]
                wr_jidan = result_jidan["winrate"]

                tqdm.write(
                    f"Ep {total_ep:>6d} | e={epsilon:.3f} lr={lr:.2e} | "
                    f"buf={len(buffer):>6d} | "
                    f"vs_{opp_name}={wr_stage:.1%} "
                    f"vs_jidan={wr_jidan:.1%} "
                    f"(1-2: {result_jidan['finish_12']}/{result_jidan['n_games']}) | "
                    f"partner_norm={partner_norm:.4f}"
                )

                _append_metrics(run_dir, {
                    "episode": total_ep,
                    "stage": stage_idx,
                    f"wr_{opp_name}": wr_stage,
                    "wr_jidan": wr_jidan,
                    "finish_12_jidan": result_jidan["finish_12"],
                    "partner_column_l2": partner_norm,
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
                    tqdm.write(f"  New best: {wr_jidan:.1%} vs jidan → {best_path}")

                # Stage gate check
                if gate is not None and wr_stage >= gate:
                    tqdm.write(
                        f"  Stage gate reached: {wr_stage:.1%} >= {gate:.0%}. "
                        f"Advancing."
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

        else:
            pbar.close()

    # Final save
    final_path = run_dir / "model_final.pt"
    torch.save({
        "lead": q_lead.state_dict(),
        "follow": q_follow.state_dict(),
        "episode": total_ep,
        "d_state": STATE_DIM_TIER1,
        "best_wr_jidan": best_wr_jidan,
    }, final_path)
    print(f"\nTraining complete. Best vs jidan: {best_wr_jidan:.1%}")
    print(f"Final checkpoint: {final_path}")
