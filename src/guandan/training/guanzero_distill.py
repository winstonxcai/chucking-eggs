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
    verbose: bool = True,
) -> int:
    """Pre-fill replay buffer with Jidan self-play trajectories.

    Runs Jidan (all 4 seats), encodes each move with GuanZero encoding,
    and pushes (non_history, history, hist_len, action_enc, mc_return) into
    the buffer. No gradient steps — pure buffer filling.

    This is the MSE-compatible replacement for cross-entropy distillation.
    The buffer will contain high-quality Jidan demonstrations expressed in
    the same ±1–3 return scale used by self-play MSE loss.

    Args:
        teacher:    JidanBot (or any agent) to generate demonstrations.
        buffer:     GuanZero ReplayBuffer to fill.
        n_games:    Number of games to simulate.
        level_rank: Game level rank.
        verbose:    Print progress every 500 games.

    Returns:
        Total number of transitions pushed to the buffer.
    """
    env = GuanDanEnv(level_rank=level_rank)
    total_trans = 0

    for game in range(n_games):
        env.reset()
        # Record (encoding, action) per player per move
        game_transitions: dict[int, list[tuple]] = {p: [] for p in range(4)}

        while not env.done:
            player = env.current_player
            legal = env.legal_moves()
            action = teacher.act(env, player)

            # Encode state + chosen action
            base = encode_base_state(env, player, level_rank)
            behavior = compute_behavior_flags(env, player, action, legal)
            nh = np.concatenate([base, behavior])          # [1075]
            hist = encode_history(env, player)             # [5, 432]
            hl = _count_valid_history_steps(env, player)
            a_enc = combo_to_108(action)                   # [108]

            game_transitions[player].append((nh, hist, hl, a_enc))
            env.step(action)

        # Assign MC returns from final game outcome
        rewards = env.get_rewards()
        for player, tlist in game_transitions.items():
            mc_return = float(rewards[player])
            for nh, hist, hl, a_enc in tlist:
                buffer.push(nh, hist, hl, a_enc, mc_return)
                total_trans += 1

        if verbose and (game + 1) % 500 == 0:
            trans_per_game = total_trans / (game + 1)
            print(f"  [Prefill] {game + 1}/{n_games} games, "
                  f"{total_trans} transitions ({trans_per_game:.1f}/game)  "
                  f"buf={len(buffer)}")

    return total_trans
