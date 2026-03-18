"""Fuzz test: verify Rust and Python movegen produce identical results."""

import random

import pytest

from guandan.cards import Card, Rank
from guandan.combos import _py_generate_all_leads
from guandan.game import GuanDanEnv

try:
    from guandan_rs import generate_all_leads as rs_leads
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


def _combo_set_py(combos):
    """Convert Python combo list to normalized set (ignoring deck ID)."""
    result = set()
    for c in combos:
        cards = tuple(sorted((x.rank, x.suit) for x in c.cards))
        result.add((int(c.type), c.key, cards))
    return result


def _combo_set_rs(combos):
    """Convert Rust combo tuples to normalized set (ignoring deck ID)."""
    result = set()
    for c in combos:
        cards = tuple(sorted((r, s) for r, s, d in c[2]))
        result.add((c[0], c[1], cards))
    return result


@pytest.mark.skipif(not HAS_RUST, reason="guandan_rs not installed")
def test_rust_matches_python_1000_games():
    """Verify Rust and Python produce identical legal move sets over 1000 games."""
    env = GuanDanEnv()
    mismatches = 0
    total_checks = 0

    for game_idx in range(1000):
        env.reset()
        while not env.done:
            hand = env.hands[env.current_player]
            hand_tuples = [(c.rank, c.suit, c.deck) for c in hand]

            py_result = _py_generate_all_leads(hand, env.level_rank)
            rs_result = rs_leads(hand_tuples, env.level_rank)

            py_set = _combo_set_py(py_result)
            rs_set = _combo_set_rs(rs_result)

            total_checks += 1
            if py_set != rs_set:
                mismatches += 1
                only_py = py_set - rs_set
                only_rs = rs_set - py_set
                if mismatches <= 5:
                    print(f"\nMismatch game={game_idx} hand_size={len(hand)} "
                          f"level_rank={env.level_rank}")
                    print(f"  Only in Python ({len(only_py)}): {list(only_py)[:3]}")
                    print(f"  Only in Rust ({len(only_rs)}): {list(only_rs)[:3]}")

            env.step(random.choice(env.legal_moves()))

    assert mismatches == 0, (
        f"{mismatches}/{total_checks} mismatches found across 1000 games"
    )
