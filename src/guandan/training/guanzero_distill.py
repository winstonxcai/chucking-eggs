"""GuanZero distillation: supervised training from Jidan (or any expert) teacher.

Stage 1: Jidan self-play (all 4 seats) — learns base policy.
Stage 2: Jidan (seats 0,2) vs opponent (seats 1,3) — learns exploitation.

Cross-entropy loss: Q-values as logits, expert's chosen action as label.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..cards import Rank
from ..game import GuanDanEnv
from .guanzero_encoding import (
    combo_to_108,
    compute_behavior_flags,
    encode_base_state,
    encode_history,
    _count_valid_history_steps,
)


def _find_action_index(legal: list, chosen) -> int | None:
    """Return index of chosen combo in legal list, matched by card set."""
    chosen_cards = frozenset(chosen.cards)
    for i, m in enumerate(legal):
        if frozenset(m.cards) == chosen_cards:
            return i
    return None


def generate_distill_data(
    teacher,
    n_games: int,
    level_rank: int,
    opponent=None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Generate supervised training data from teacher play.

    Args:
        teacher:    Agent that acts as the expert (e.g. JidanBot).
        n_games:    Number of games to simulate.
        level_rank: Current level rank (e.g. Rank.TWO).
        opponent:   If None, teacher plays all 4 seats (self-play).
                    If provided, teacher plays seats {0, 2}; opponent plays {1, 3}.
        verbose:    Print progress every 500 games.

    Returns:
        List of decision dicts, each with:
          'non_histories': list of B np.ndarray [1075]  — one per legal action
          'history':       np.ndarray [5, 432]
          'hist_len':      int
          'action_encs':   list of B np.ndarray [108]
          'expert_idx':    int — which legal action the teacher chose
    """
    env = GuanDanEnv(level_rank=level_rank)
    data: list[dict[str, Any]] = []

    for game in range(n_games):
        env.reset()

        while not env.done:
            player = env.current_player
            legal = env.legal_moves()

            is_teacher_seat = (opponent is None) or (player in (0, 2))

            if not is_teacher_seat:
                env.step(opponent.act(env, player))
                continue

            # Skip trivial decisions (only one legal move)
            if len(legal) <= 1:
                env.step(legal[0])
                continue

            # Teacher chooses
            expert_action = teacher.act(env, player)
            expert_idx = _find_action_index(legal, expert_action)
            if expert_idx is None:
                # Teacher returned an illegal move — fall back to first legal
                env.step(legal[0])
                continue

            # Encode ALL candidate actions (behavior flags differ per action)
            base = encode_base_state(env, player, level_rank)  # [1066]
            history = encode_history(env, player)              # [5, 432]
            hist_len = _count_valid_history_steps(env, player)

            non_histories = []
            action_encs = []
            for m in legal:
                behavior = compute_behavior_flags(env, player, m, legal)  # [9]
                nh = np.concatenate([base, behavior])                      # [1075]
                non_histories.append(nh)
                action_encs.append(combo_to_108(m))                        # [108]

            data.append({
                "non_histories": non_histories,
                "history": history,
                "hist_len": hist_len,
                "action_encs": action_encs,
                "expert_idx": expert_idx,
            })

            env.step(expert_action)

        if verbose and (game + 1) % 500 == 0:
            print(f"  {game + 1}/{n_games} games, {len(data)} decisions "
                  f"({len(data) / (game + 1):.1f} decisions/game)")

    return data


def train_distill_epoch(
    net,
    optimizer: torch.optim.Optimizer,
    data: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = 32,
) -> tuple[float, float]:
    """One epoch of cross-entropy distillation.

    Uses gradient accumulation because #legal moves varies per decision.

    Returns:
        (avg_loss, top1_accuracy)
    """
    random.shuffle(data)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    accum_count = 0

    optimizer.zero_grad()

    for d in data:
        B = len(d["action_encs"])
        if B == 0:
            continue

        nh = torch.tensor(
            np.array(d["non_histories"]), dtype=torch.float32, device=device
        )  # [B, 1075]
        hist = torch.tensor(
            d["history"], dtype=torch.float32, device=device
        )  # [5, 432]
        hl = torch.tensor([d["hist_len"]], dtype=torch.long, device=device)  # [1]
        act = torch.tensor(
            np.array(d["action_encs"]), dtype=torch.float32, device=device
        )  # [B, 108]

        q_vals = net.forward_fast(nh, hist, hl, act)  # [B]

        target = torch.tensor(d["expert_idx"], dtype=torch.long, device=device)
        # q_vals as logits over B actions; target = expert's choice
        loss = F.cross_entropy(q_vals.unsqueeze(0), target.unsqueeze(0))

        (loss / batch_size).backward()
        accum_count += 1

        total_loss += loss.item()
        total_correct += int(q_vals.argmax().item() == d["expert_idx"])
        total_count += 1

        if accum_count >= batch_size:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0

    # Flush remaining gradients
    if accum_count > 0:
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / max(total_count, 1)
    accuracy = total_correct / max(total_count, 1)
    return avg_loss, accuracy


def run_distillation(
    net,
    device: torch.device,
    level_rank: int = Rank.TWO,
    distill_games: int = 5000,
    distill_epochs: int = 3,
    vs_games: int = 3000,
    vs_epochs: int = 3,
    save_path: Path | None = None,
    verbose: bool = True,
) -> None:
    """Full distillation pipeline: Stage 1 (self-play) + Stage 2 (vs opponent).

    Args:
        net:           GuanZeroNetwork instance.
        device:        torch device.
        level_rank:    Level rank for the game.
        distill_games: Number of Jidan self-play games (Stage 1).
        distill_epochs: Epochs to train on Stage 1 data.
        vs_games:      Number of Jidan vs Strategic games (Stage 2).
        vs_epochs:     Epochs to train on Stage 2 data.
        save_path:     Where to save the distilled checkpoint.
        verbose:       Print progress.
    """
    from ..agents.jidan_bot import JidanBot
    from ..agents.strategic_bot import StrategicBot

    teacher = JidanBot()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    # Stage 1: Jidan self-play
    if distill_games > 0:
        if verbose:
            print(f"[Distill] Stage 1: generating {distill_games} Jidan self-play games...")
        data1 = generate_distill_data(teacher, distill_games, level_rank, verbose=verbose)
        if verbose:
            print(f"[Distill] Stage 1: {len(data1)} decisions, training {distill_epochs} epochs...")

        net.train()
        for epoch in range(distill_epochs):
            loss, acc = train_distill_epoch(net, optimizer, data1, device)
            if verbose:
                print(f"  Epoch {epoch + 1}/{distill_epochs}: loss={loss:.4f} acc={acc:.3f}")
        del data1

    # Stage 2: Jidan vs Strategic
    if vs_games > 0:
        if verbose:
            print(f"[Distill] Stage 2: generating {vs_games} Jidan vs Strategic games...")
        opponent = StrategicBot()
        data2 = generate_distill_data(teacher, vs_games, level_rank, opponent=opponent, verbose=verbose)
        if verbose:
            print(f"[Distill] Stage 2: {len(data2)} decisions, training {vs_epochs} epochs (lr=5e-5)...")

        for param_group in optimizer.param_groups:
            param_group["lr"] = 5e-5

        net.train()
        for epoch in range(vs_epochs):
            loss, acc = train_distill_epoch(net, optimizer, data2, device)
            if verbose:
                print(f"  Epoch {epoch + 1}/{vs_epochs}: loss={loss:.4f} acc={acc:.3f}")
        del data2

    # Save checkpoint
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": net.state_dict(), "level_rank": level_rank}, save_path)
        if verbose:
            print(f"[Distill] Saved to {save_path}")
