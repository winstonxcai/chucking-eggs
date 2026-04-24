"""Tests for BeliefModel — pass-derived hard constraints + rank counting."""

from guandan.agents.belief import BeliefModel, OppConstraints
from guandan.cards import Card, ComboType, Rank, Suit
from guandan.combos import Combo
from guandan.game import GuanDanEnv


def _fresh_env() -> GuanDanEnv:
    env = GuanDanEnv(level_rank=Rank.TWO)
    env.reset(seed=42)
    return env


def test_fresh_env_no_constraints():
    """On a fresh env with no moves yet, every opp's constraints are empty."""
    env = _fresh_env()
    bm = BeliefModel()
    cs = bm.constraints(env)
    for seat in range(4):
        c = cs[seat]
        assert c.max_single_key is None
        assert c.max_pair_key is None
        assert c.max_triple_key is None
        assert c.no_bomb is False


def test_rank_distribution_fresh():
    """Fresh env: opposition unknowns = 8 - own[rank] - partner[rank] for player 0."""
    env = _fresh_env()
    bm = BeliefModel()
    rd = bm.rank_distribution(env, 0)

    # Build expected counts manually
    from collections import Counter
    own = Counter(c.rank for c in env.hands[0])
    partner = Counter(c.rank for c in env.hands[2])
    for rank in range(2, 15):
        expected = 8 - own.get(rank, 0) - partner.get(rank, 0)
        assert rd.get(rank, 0) == expected, f"rank {rank}: got {rd.get(rank, 0)}, want {expected}"
    for rank in (Rank.BLACK_JOKER, Rank.RED_JOKER):
        expected = 2 - own.get(rank, 0) - partner.get(rank, 0)
        assert rd.get(rank, 0) == expected


def test_pass_on_single_excludes_higher_singles():
    """Scripted scenario: player 1 passes on a single-7 → max_single_key[1] = 7."""
    env = _fresh_env()
    # Inject a fake history: player 0 led single-7, player 1 passed.
    seven = Card(rank=7, suit=Suit.SPADE, deck=0)
    single_7 = Combo(ComboType.SINGLE, key_rank=7, cards=[seven])
    env.move_history = [(0, single_7), (1, Combo(ComboType.PASS, 0, []))]

    bm = BeliefModel()
    c = bm.constraints(env)[1]
    assert c.max_single_key == 7
    assert c.no_bomb is True   # passed on a non-bomb trick → no bomb either
    assert c.max_pair_key is None  # untouched
    assert c.max_triple_key is None


def test_violates_single_constraint():
    """A hand containing a card with rank > max_single_key violates."""
    c = OppConstraints(max_single_key=7)  # no single > 7 (level order)
    # An 8 is rank 8 > 7 → violates
    eight = Card(rank=8, suit=Suit.SPADE, deck=0)
    assert c.violates({eight}, level_rank=Rank.TWO)
    # A 6 is rank 6 < 7 → OK
    six = Card(rank=6, suit=Suit.SPADE, deck=0)
    assert not c.violates({six}, level_rank=Rank.TWO)


def test_level_card_treated_above_ace():
    """With level_rank=2, a 2-card has level_order_key=15, above A=14."""
    c = OppConstraints(max_single_key=14)  # passed on single-A
    # Level card (rank 2 with level=2) has level_order_key 15 > 14 → violates
    two = Card(rank=2, suit=Suit.SPADE, deck=0)
    assert c.violates({two}, level_rank=Rank.TWO)
    # A King (rank 13, key 13) → OK
    king = Card(rank=13, suit=Suit.SPADE, deck=0)
    assert not c.violates({king}, level_rank=Rank.TWO)


def test_pair_constraint():
    """no_pair_above K: any rank R > K appearing twice violates."""
    c = OppConstraints(max_pair_key=10)  # passed on pair-10
    eight = Card(rank=8, suit=Suit.SPADE, deck=0)
    eight2 = Card(rank=8, suit=Suit.HEART, deck=1)
    # Pair of 8s, key 8 < 10 → OK
    assert not c.violates({eight, eight2}, level_rank=Rank.TWO)
    jack = Card(rank=11, suit=Suit.SPADE, deck=0)
    jack2 = Card(rank=11, suit=Suit.HEART, deck=1)
    # Pair of Js, key 11 > 10 → violates
    assert c.violates({jack, jack2}, level_rank=Rank.TWO)
    # Single J → OK (need a pair to violate)
    assert not c.violates({jack}, level_rank=Rank.TWO)


def test_no_bomb_constraint():
    """no_bomb: any rank with count >= 4 violates."""
    c = OppConstraints(no_bomb=True)
    # 4 sevens → violates
    sevens = {Card(rank=7, suit=s, deck=d) for s in range(4) for d in range(1)}
    assert len(sevens) == 4
    assert c.violates(sevens, level_rank=Rank.TWO)
    # 3 sevens → OK (no bomb yet)
    three_sevens = set(list(sevens)[:3])
    assert not c.violates(three_sevens, level_rank=Rank.TWO)


def test_no_bomb_4_jokers():
    """no_bomb also covers the 4-joker bomb."""
    c = OppConstraints(no_bomb=True)
    bj1 = Card(rank=Rank.BLACK_JOKER, suit=0, deck=0)
    bj2 = Card(rank=Rank.BLACK_JOKER, suit=0, deck=1)
    rj1 = Card(rank=Rank.RED_JOKER, suit=1, deck=0)
    rj2 = Card(rank=Rank.RED_JOKER, suit=1, deck=1)
    assert c.violates({bj1, bj2, rj1, rj2}, level_rank=Rank.TWO)
    # 2 BJ + 1 RJ → OK
    assert not c.violates({bj1, bj2, rj1}, level_rank=Rank.TWO)


def test_pass_on_bomb_does_not_set_no_bomb():
    """If trick was a bomb and opp passed, that doesn't tell us they have no bomb
    (they may have a smaller bomb that can't beat). For Phase 1a we just don't
    set no_bomb in this case (skipping the bomb-vs-bomb constraint)."""
    env = _fresh_env()
    seven = Card(rank=7, suit=Suit.SPADE, deck=0)
    bomb_7 = Combo(ComboType.BOMB_4, key_rank=7,
                   cards=[seven, Card(7, 1, 0), Card(7, 2, 0), Card(7, 3, 0)])
    env.move_history = [(0, bomb_7), (1, Combo(ComboType.PASS, 0, []))]

    bm = BeliefModel()
    c = bm.constraints(env)[1]
    assert c.no_bomb is False  # not constrained: trick was a bomb


def test_pass_with_no_trick_state_skipped():
    """A pass with no current trick (shouldn't normally happen but defensive)."""
    env = _fresh_env()
    env.move_history = [(1, Combo(ComboType.PASS, 0, []))]  # pass with no trick
    bm = BeliefModel()
    c = bm.constraints(env)[1]
    # Nothing should be constrained
    assert c.max_single_key is None
    assert c.no_bomb is False


def test_constraints_tighten_with_more_passes():
    """Multiple passes by same opp tighten constraints to the smallest key."""
    env = _fresh_env()
    env.move_history = [
        # Trick 1: single-K, opp 1 passes → max_single_key[1] = K=13
        (0, Combo(ComboType.SINGLE, 13, [Card(13, 0, 0)])),
        (1, Combo(ComboType.PASS, 0, [])),
        # Trick 2: single-7, opp 1 passes → max_single_key[1] = min(13, 7) = 7 (tighter)
        (0, Combo(ComboType.SINGLE, 7, [Card(7, 0, 0)])),
        (1, Combo(ComboType.PASS, 0, [])),
    ]
    bm = BeliefModel()
    c = bm.constraints(env)[1]
    assert c.max_single_key == 7
