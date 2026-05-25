from __future__ import annotations

import builtins
import importlib.util
import random

import guandan_rs

from guandan.cards import Card, ComboType, Rank, Suit
from guandan.combos import Combo
from guandan.dart.utils.legal_utils import dedup_strategic, select_legal
from guandan.game import GuanDanEnv


def _combo(ctype: ComboType, key: int, cards: list[Card], length: int = 0) -> Combo:
    return Combo(ctype, key, cards, length=length)


def _strategic_sig(combo: Combo) -> tuple:
    combo_type = int(combo.type)
    out = [combo_type, int(combo.key), int(combo.length), int(combo.wild_count)]
    if 1 <= combo_type <= 9 or 11 <= combo_type <= 15:
        out.extend(sorted(int(card.rank) for card in combo.cards))
    else:
        for card in sorted(combo.cards, key=lambda c: (int(c.rank), int(c.suit))):
            out.extend([int(card.rank), int(card.suit)])
    return tuple(out)


def _tuple_sig(combo_tuple: tuple) -> tuple:
    combo_type, key, cards, length, wild_count = combo_tuple
    if 1 <= combo_type <= 9 or 11 <= combo_type <= 15:
        card_sig = tuple(sorted(rank for rank, _suit, _deck in cards))
    else:
        card_sig = tuple(sorted((rank, suit) for rank, suit, _deck in cards))
    return (
        combo_type,
        key,
        card_sig,
        length,
        wild_count,
    )


def _legacy_select_legal(env: GuanDanEnv, player: int) -> list[Combo]:
    legal = dedup_strategic(env.legal_moves(player))
    if not env.is_leading() and not any(m.type == ComboType.PASS for m in legal):
        legal.append(Combo(ComboType.PASS, 0, [], 0, 0))
    return legal


def _load_guandan_rs_without_native(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "_guandan_rs":
            raise ImportError("forced fallback")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    spec = importlib.util.spec_from_file_location(
        "_guandan_rs_fallback_test",
        guandan_rs.__file__,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dedup_strategic_collapses_suit_variants_for_ordinary_moves():
    first = _combo(
        ComboType.PAIR,
        Rank.ACE,
        [Card(Rank.ACE, Suit.SPADE, 0), Card(Rank.ACE, Suit.HEART, 0)],
    )
    second = _combo(
        ComboType.PAIR,
        Rank.ACE,
        [Card(Rank.ACE, Suit.DIAMOND, 0), Card(Rank.ACE, Suit.CLUB, 0)],
    )

    out = dedup_strategic([first, second])
    assert len(out) == 1
    assert out[0].type == first.type and out[0].key == first.key


def test_dedup_strategic_preserves_pass_and_suit_sensitive_bombs():
    pass_move = _combo(ComboType.PASS, 0, [])
    spade_sf = _combo(
        ComboType.STRAIGHT_FLUSH,
        Rank.SIX,
        [Card(rank, Suit.SPADE, 0) for rank in range(Rank.TWO, Rank.SEVEN)],
        length=5,
    )
    heart_sf = _combo(
        ComboType.STRAIGHT_FLUSH,
        Rank.SIX,
        [Card(rank, Suit.HEART, 0) for rank in range(Rank.TWO, Rank.SEVEN)],
        length=5,
    )
    joker_bomb = _combo(
        ComboType.BOMB_JOKER,
        99,
        [
            Card(Rank.BLACK_JOKER, 0, 0),
            Card(Rank.BLACK_JOKER, 0, 1),
            Card(Rank.RED_JOKER, 1, 0),
            Card(Rank.RED_JOKER, 1, 1),
        ],
    )

    out = dedup_strategic([pass_move, spade_sf, heart_sf, joker_bomb])
    assert len(out) == 4
    assert [c.type for c in out] == [pass_move.type, spade_sf.type, heart_sf.type, joker_bomb.type]


def test_select_legal_matches_legacy_path_over_random_rollout():
    rng = random.Random(20240520)
    env = GuanDanEnv(seed=20240520)
    seen_lead = False
    seen_response = False

    for _ in range(400):
        if env.done:
            break
        player = env.current_player
        seen_lead = seen_lead or env.is_leading()
        seen_response = seen_response or not env.is_leading()

        actual = select_legal(env, player)
        expected = _legacy_select_legal(env, player)
        assert {_strategic_sig(c) for c in actual} == {
            _strategic_sig(c) for c in expected
        }

        env.step(actual[rng.randrange(len(actual))])

    assert seen_lead
    assert seen_response


def test_guandan_rs_python_fallback_matches_public_select_legal(monkeypatch):
    fallback = _load_guandan_rs_without_native(monkeypatch)
    assert fallback.HAS_NATIVE is False

    rng = random.Random(20240521)
    env = GuanDanEnv(seed=20240521)
    for _ in range(100):
        if env.done:
            break
        player = env.current_player
        hand = [(c.rank, c.suit, c.deck) for c in env.hands[player]]
        trick = None if env.current_trick is None else (
            int(env.current_trick.type),
            env.current_trick.key,
            [(c.rank, c.suit, c.deck) for c in env.current_trick.cards],
            env.current_trick.length,
            env.current_trick.wild_count,
        )

        actual = fallback.select_legal(hand, int(env.level_rank), trick)
        expected = guandan_rs.select_legal(hand, int(env.level_rank), trick)
        assert {_tuple_sig(t) for t in actual} == {_tuple_sig(t) for t in expected}

        legal = select_legal(env, player)
        env.step(legal[rng.randrange(len(legal))])
