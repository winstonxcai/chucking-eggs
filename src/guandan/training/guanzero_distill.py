"""GuanZero buffer prefill from Jidan teacher (MSE-compatible).

Pre-fills the replay buffer with Jidan's trajectories using the same
(state, action, mc_return) format and MSE loss as self-play. This is
equivalent to pretrain_from_heuristic in train.py but using the GuanZero
108-dim encoding and Jidan as the teacher.

No separate training phase — only buffer filling. Self-play then starts
from a warm buffer of high-quality Jidan demonstrations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..cards import Rank
from ..game import GuanDanEnv
from .guanzero_encoding import (
    _count_valid_history_steps,
    combo_to_108,
    compute_behavior_flags,
    encode_base_state,
    encode_history,
)


def prefill_buffer_from_jidan(
    teacher,
    buffer,
    n_games: int,
    level_rank: int = Rank.TWO,
    net=None,
    optimizer=None,
    device=None,
    batch_size: int = 512,
    train_steps_per_game: int = 2,
    verbose: bool = True,
) -> int:
    """Pre-fill replay buffer with Jidan trajectories and optionally train.

    Runs Jidan (all 4 seats), encodes each move with GuanZero encoding,
    pushes (non_history, history, hist_len, action_enc, mc_return) into the
    buffer, then optionally runs `train_steps_per_game` MSE gradient steps
    per game — same pattern as pretrain_from_heuristic in train.py.

    Args:
        teacher:              JidanBot (or any agent) to generate demonstrations.
        buffer:               GuanZero ReplayBuffer to fill.
        n_games:              Number of games to simulate.
        level_rank:           Game level rank.
        net:                  GuanZeroNetwork to train (None = buffer-fill only).
        optimizer:            Adam optimizer (required if net is not None).
        batch_size:           Training batch size.
        train_steps_per_game: Gradient steps per game (default: 2).
        verbose:              Print progress every 500 games.

    Returns:
        Total number of transitions pushed to the buffer.
    """
    from .guanzero_selfplay import train_dmc_step

    env = GuanDanEnv(level_rank=level_rank)
    total_trans = 0

    for game in range(n_games):
        env.reset()
        game_transitions: dict[int, list[tuple]] = {p: [] for p in range(4)}

        while not env.done:
            player = env.current_player
            legal = env.legal_moves()
            action = teacher.act(env, player)

            base = encode_base_state(env, player, level_rank)
            behavior = compute_behavior_flags(env, player, action, legal)
            nh = np.concatenate([base, behavior])
            hist = encode_history(env, player)
            hl = _count_valid_history_steps(env, player)
            a_enc = combo_to_108(action)

            game_transitions[player].append((nh, hist, hl, a_enc))
            env.step(action)

        rewards = env.get_rewards()
        for player, tlist in game_transitions.items():
            mc_return = float(rewards[player])
            for nh, hist, hl, a_enc in tlist:
                buffer.push(nh, hist, hl, a_enc, mc_return)
                total_trans += 1

        # Train on buffer after each game
        if net is not None and optimizer is not None:
            for _ in range(train_steps_per_game):
                train_dmc_step(net, optimizer, buffer, batch_size, device)

        if verbose and (game + 1) % 500 == 0:
            trans_per_game = total_trans / (game + 1)
            print(f"  [Prefill] {game + 1}/{n_games} games, "
                  f"{total_trans} transitions ({trans_per_game:.1f}/game)  "
                  f"buf={len(buffer)}")

    return total_trans
