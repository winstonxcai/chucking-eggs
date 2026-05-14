"""Parity test: Rust encoder output == Python RoleAwareStateActionEncoder.

For N random game states, asserts that encode_state_for_parity() (Rust) and
RoleAwareStateActionEncoder._encode_state() (Python) produce identical f32 arrays
within tolerance 1e-5.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import guandan_rs
from guandan.cards import card_to_id, ComboType
from guandan.game import GuanDanEnv
from guandan.guanzero.model.encoding.role_encoder import (
    ROLE_ENCODE_STATE_KEYS,
    ROLE_ENCODE_CHANNEL_SHAPES,
    RoleAwareStateActionEncoder,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _last_nonpass_per_seat(move_history, n_seats: int = 4) -> np.ndarray:
    """Compute last non-pass action per absolute seat from move_history."""
    out = np.zeros((n_seats, 108), dtype=np.uint8)
    seen = [False] * n_seats
    for actor, combo in reversed(move_history):
        if not seen[actor] and combo.type != ComboType.PASS:
            for card in combo.cards:
                out[actor, card_to_id(card)] = 1
            seen[actor] = True
        if all(seen):
            break
    return out


def _combo_to_pytuple(combo) -> tuple:
    cards_py = [(c.rank, c.suit, c.deck) for c in combo.cards]
    return (int(combo.type), combo.key, cards_py, combo.length, combo.wild_count)


def _call_rust_encoder(env: GuanDanEnv, player: int, legal: list) -> np.ndarray:
    """Call Rust encode_state_for_parity and return a float32 numpy array."""
    last_nonpass = _last_nonpass_per_seat(env.move_history)
    history_py = [(seat, _combo_to_pytuple(c)) for seat, c in env.move_history]
    legal_py = [_combo_to_pytuple(c) for c in legal]
    trick_py = _combo_to_pytuple(env.current_trick) if env.current_trick else None
    buf: bytes = guandan_rs.encode_state_for_parity(
        hand_multihot=env.hand_multihot.tolist(),
        played_multihot=env.played_multihot.tolist(),
        last_nonpass_multihot=last_nonpass.tolist(),
        bombs_played=env.bombs_played.tolist(),
        hand_sizes=[len(env.hands[s]) for s in range(4)],
        is_out=list(env.is_out),
        current_trick=trick_py,
        trick_winner=env.trick_winner,
        level_rank=env.level_rank,
        move_history=history_py,
        player=player,
        legal_moves=legal_py,
    )
    return np.frombuffer(buf, dtype=np.float32)


def _flatten_python_state(enc_dict: dict) -> np.ndarray:
    """Flatten the Python encoder's state dict to a float32 vector in ROLE_ENCODE_STATE_KEYS order."""
    parts = []
    for key in ROLE_ENCODE_STATE_KEYS:
        arr = enc_dict[key].astype(np.float32)
        parts.append(arr.ravel())
    return np.concatenate(parts)


# ─── Tests ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(20))
def test_encoder_parity_one_episode(seed: int):
    """Rust encoder matches Python encoder on every step of one seeded episode."""
    encoder = RoleAwareStateActionEncoder(head_scheme="absolute_seat")
    env = GuanDanEnv(level_rank=2)
    env.reset(seed=seed)

    rng = __import__("random").Random(seed)
    mismatches: list[str] = []

    while not env.done:
        player = env.current_player
        legal = env.legal_moves()

        py_state = encoder._encode_state(env, player, legal)
        py_flat = _flatten_python_state(py_state)

        rust_flat = _call_rust_encoder(env, player, legal)

        if not np.allclose(py_flat, rust_flat, atol=1e-5):
            diff_idx = np.where(~np.isclose(py_flat, rust_flat, atol=1e-5))[0]
            mismatches.append(
                f"seed={seed} step: player={player} "
                f"n_diffs={len(diff_idx)} "
                f"max_diff={np.abs(py_flat[diff_idx] - rust_flat[diff_idx]).max():.6f}"
            )

        # Advance with a random move (same for both)
        action = rng.choice(legal)
        env.step(action)

    assert not mismatches, "\n".join(mismatches)


def test_encoder_parity_state_dim():
    """Rust encoder always returns STATE_DIM * 4 bytes."""
    STATE_DIM = sum(
        math.prod(ROLE_ENCODE_CHANNEL_SHAPES[k]) if ROLE_ENCODE_CHANNEL_SHAPES[k] else 1
        for k in ROLE_ENCODE_STATE_KEYS
    )
    env = GuanDanEnv()
    env.reset(seed=0)
    player = env.current_player
    legal = env.legal_moves()
    buf = _call_rust_encoder(env, player, legal)
    assert buf.size == STATE_DIM, f"expected {STATE_DIM} floats, got {buf.size}"
