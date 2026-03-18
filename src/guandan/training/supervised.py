"""Supervised distillation: train Q-networks by imitating teacher bots."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F

from ..game import GuanDanEnv
from .encoding import encode_action, encode_history, encode_state


def generate_teacher_data(teacher, n_games: int, level_rank: int) -> list[dict]:
    """Run teacher self-play and record every non-trivial decision.

    Returns list of dicts with: state, action_encs, history, hist_len,
    expert_idx, is_leading.
    """
    env = GuanDanEnv(level_rank)
    data: list[dict] = []

    for game_idx in range(n_games):
        env.reset()

        while not env.done:
            player = env.current_player
            legal = env.legal_moves(player)

            if len(legal) <= 1:
                env.step(legal[0])
                continue

            expert_action = teacher.act(env, player)
            expert_idx = _find_action_index(legal, expert_action)
            if expert_idx is None:
                env.step(expert_action)
                continue

            state_enc = encode_state(env, player)
            hand = env.hands[player]
            action_encs = [encode_action(m, hand, level_rank) for m in legal]
            history, hist_len = encode_history(env, player, level_rank)
            is_leading = env.current_trick is None

            data.append({
                "state": state_enc,
                "action_encs": action_encs,
                "history": history,
                "hist_len": hist_len,
                "expert_idx": expert_idx,
                "is_leading": is_leading,
            })

            env.step(expert_action)

        if (game_idx + 1) % 500 == 0:
            print(
                f"  Generated {game_idx + 1}/{n_games} games, "
                f"{len(data)} decision points"
            )

    print(
        f"  Total: {len(data)} decision points from {n_games} games "
        f"({len(data) / n_games:.0f} per game)"
    )
    return data


def _find_action_index(legal_moves, chosen_action) -> int | None:
    """Find index of chosen_action in legal_moves by matching type + cards."""
    chosen_cards = {(c.rank, c.suit, c.deck) for c in chosen_action.cards}
    chosen_type = chosen_action.type
    for i, m in enumerate(legal_moves):
        if m.type != chosen_type:
            continue
        if {(c.rank, c.suit, c.deck) for c in m.cards} == chosen_cards:
            return i
    return None


def train_supervised(
    q_lead,
    q_follow,
    opt_lead,
    opt_follow,
    teacher_data: list[dict],
    epochs: int = 3,
    batch_size: int = 32,
    device: str = "mps",
) -> None:
    """Train Q-networks via cross-entropy on teacher decisions."""
    for epoch in range(epochs):
        random.shuffle(teacher_data)

        lead_losses, follow_losses = [], []
        lead_correct, lead_total = 0, 0
        follow_correct, follow_total = 0, 0

        opt_lead.zero_grad()
        opt_follow.zero_grad()
        lead_batch_count = 0
        follow_batch_count = 0

        for d in teacher_data:
            q_net = q_lead if d["is_leading"] else q_follow

            B = len(d["action_encs"])
            s = (
                torch.tensor(d["state"], dtype=torch.float32)
                .unsqueeze(0)
                .expand(B, -1)
                .to(device)
            )
            a = torch.tensor(
                np.array(d["action_encs"]), dtype=torch.float32
            ).to(device)
            h = (
                torch.tensor(d["history"], dtype=torch.float32)
                .unsqueeze(0)
                .expand(B, -1, -1)
                .to(device)
            )
            hl = torch.tensor([d["hist_len"]], dtype=torch.long).expand(B)

            q_values = q_net(s, a, h, hl)  # [B]

            target = torch.tensor(d["expert_idx"], dtype=torch.long, device=device)
            loss = F.cross_entropy(q_values.unsqueeze(0), target.unsqueeze(0))

            (loss / batch_size).backward()

            predicted = q_values.argmax().item()
            if d["is_leading"]:
                lead_losses.append(loss.item())
                lead_correct += predicted == d["expert_idx"]
                lead_total += 1
                lead_batch_count += 1
                if lead_batch_count >= batch_size:
                    torch.nn.utils.clip_grad_norm_(q_lead.parameters(), max_norm=1.0)
                    opt_lead.step()
                    opt_lead.zero_grad()
                    lead_batch_count = 0
            else:
                follow_losses.append(loss.item())
                follow_correct += predicted == d["expert_idx"]
                follow_total += 1
                follow_batch_count += 1
                if follow_batch_count >= batch_size:
                    torch.nn.utils.clip_grad_norm_(
                        q_follow.parameters(), max_norm=1.0
                    )
                    opt_follow.step()
                    opt_follow.zero_grad()
                    follow_batch_count = 0

        # Flush remaining gradients
        if lead_batch_count > 0:
            torch.nn.utils.clip_grad_norm_(q_lead.parameters(), max_norm=1.0)
            opt_lead.step()
            opt_lead.zero_grad()
        if follow_batch_count > 0:
            torch.nn.utils.clip_grad_norm_(q_follow.parameters(), max_norm=1.0)
            opt_follow.step()
            opt_follow.zero_grad()

        lead_acc = lead_correct / lead_total if lead_total > 0 else 0
        follow_acc = follow_correct / follow_total if follow_total > 0 else 0
        lead_avg = sum(lead_losses) / len(lead_losses) if lead_losses else 0
        follow_avg = sum(follow_losses) / len(follow_losses) if follow_losses else 0

        print(
            f"  Epoch {epoch + 1}/{epochs}: "
            f"lead_loss={lead_avg:.4f} lead_acc={lead_acc:.1%} ({lead_total}) | "
            f"follow_loss={follow_avg:.4f} follow_acc={follow_acc:.1%} ({follow_total})"
        )
