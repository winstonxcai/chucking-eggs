"""GuanZero buffer prefill from Jidan teacher (hybrid encoding, MSE-compatible).

Pre-fills the replay buffer with Jidan's trajectories using the hybrid
417+9 state encoding and existing 160-dim action encoding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..cards import Rank
from ..game import GuanDanEnv
from .guanzero_encoding import (
    encode_state_hybrid,
    encode_history_padded,
    _encode_action_160,
)


def prefill_buffer_from_jidan(
    teacher,
    buffers: "list | ReplayBuffer",
    n_games: int,
    level_rank: int = Rank.TWO,
    nets: "list | None" = None,
    optimizers: "list | None" = None,
    device=None,
    batch_size: int = 512,
    train_steps_per_game: int = 2,
    verbose: bool = True,
) -> int:
    """Pre-fill 4 seat-specific replay buffers from Jidan trajectories.

    Accepts either:
    - `buffers`: a list of 4 ReplayBuffers (one per seat) — routes each
      transition to the corresponding seat's buffer.
    - `buffers`: a single ReplayBuffer — all transitions go to it (legacy).

    Optionally trains `nets[seat]` on `buffers[seat]` after each game.

    Args:
        teacher:              Agent to generate demonstrations (e.g. JidanBot).
        buffers:              List of 4 ReplayBuffers, or a single ReplayBuffer.
        n_games:              Number of games to simulate.
        level_rank:           Game level rank.
        nets:                 List of 4 GuanZeroNetworks (or None = no training).
        optimizers:           List of 4 Adam optimizers (required if nets given).
        device:               torch.device.
        batch_size:           Training batch size.
        train_steps_per_game: Gradient steps per seat per game.
        verbose:              Print progress every 500 games.

    Returns:
        Total transitions pushed across all buffers.
    """
    from .guanzero_selfplay import train_dmc_step

    # Normalise: if single buffer passed, wrap in list of 4 (broadcast)
    single_buffer = not isinstance(buffers, list)
    buf_list = [buffers] * 4 if single_buffer else buffers

    env = GuanDanEnv(level_rank=level_rank)
    total_trans = 0

    for game in range(n_games):
        env.reset()
        game_transitions: dict[int, list[tuple]] = {p: [] for p in range(4)}

        while not env.done:
            player = env.current_player
            legal = env.legal_moves()
            action = teacher.act(env, player)

            nh = encode_state_hybrid(env, player, action, legal)               # [426]
            hist, hl = encode_history_padded(env, player, level_rank)          # [15,83], int
            a_enc = _encode_action_160(action, env.hands[player], level_rank)  # [160]

            game_transitions[player].append((nh, hist, hl, a_enc))
            env.step(action)

        rewards = env.get_rewards()
        for player, tlist in game_transitions.items():
            mc_return = float(rewards[player])
            for nh, hist, hl, a_enc in tlist:
                buf_list[player].push(nh, hist, hl, a_enc, mc_return)
                total_trans += 1

        # Train each seat's network on its own buffer
        if nets is not None and optimizers is not None:
            for seat in range(4):
                for _ in range(train_steps_per_game):
                    train_dmc_step(nets[seat], optimizers[seat], buf_list[seat], batch_size, device)

        if verbose and (game + 1) % 500 == 0:
            buf_sizes = [len(b) for b in buf_list]
            trans_per_game = total_trans / (game + 1)
            print(f"  [Prefill] {game + 1}/{n_games} games, "
                  f"{total_trans} transitions ({trans_per_game:.1f}/game)  "
                  f"bufs={buf_sizes}")

    return total_trans


# Keep type hint importable without circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .guanzero_selfplay import ReplayBuffer
