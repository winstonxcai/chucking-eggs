"""Match selected card IDs to a legal Combo."""

from __future__ import annotations

from guandan.cards import BOMB_SIZE_TO_TYPE, Card, ComboType
from guandan.combos import Combo


def find_matching_combo(
    selected_card_ids: list[str],
    legal_moves: list[Combo],
    hand: set[Card] | None = None,
    level_rank: int | None = None,
) -> list[Combo]:
    """Find legal combos whose card set matches the selected cards.

    Pass 1: exact ID match (rank-suit-deck).
    Pass 2 (needs hand): deck-normalized — same (rank,suit) multiset, different deck.
    Pass 3 (needs hand): N-of-a-kind bomb — all same rank, matching bomb type exists.

    Fallback passes construct a new Combo with the player's actual card objects
    so env.step() removes the correct physical cards from the hand.
    """
    target = frozenset(selected_card_ids)
    matches = []

    # Pass 1: exact match
    for combo in legal_moves:
        combo_ids = frozenset(
            f"{c.rank}-{c.suit}-{c.deck}" for c in combo.cards
        )
        if combo_ids == target:
            matches.append(combo)

    if matches or hand is None:
        return matches

    # Build id → Card lookup from hand
    id_to_card = {f"{c.rank}-{c.suit}-{c.deck}": c for c in hand}
    selected = [id_to_card.get(cid) for cid in selected_card_ids]
    if any(c is None for c in selected):
        return []
    sel_cards: list[Card] = list(selected)  # type: ignore

    # Pass 2: deck-normalized — same (rank, suit) multiset, different deck assignment
    sel_rs = tuple(sorted((c.rank, c.suit) for c in sel_cards))
    for combo in legal_moves:
        if combo.type == ComboType.PASS:
            continue
        combo_rs = tuple(sorted((c.rank, c.suit) for c in combo.cards))
        if combo_rs == sel_rs:
            return [Combo(combo.type, combo.key, sel_cards,
                          combo.length, combo.wild_count)]

    # Pass 3: N-of-a-kind bomb — all same rank, any legal bomb of that type
    ranks = {c.rank for c in sel_cards}
    if len(ranks) == 1:
        rank = next(iter(ranks))
        n = len(sel_cards)
        bomb_type = BOMB_SIZE_TO_TYPE.get(n)
        if bomb_type:
            for combo in legal_moves:
                if combo.type == bomb_type and combo.key == rank:
                    return [Combo(combo.type, combo.key, sel_cards,
                                  combo.length, combo.wild_count)]

    # Pass 4: use game engine combo detection on the selected cards as a fallback.
    # This handles deck-copy mismatches, wildcard substitution variants, and any
    # other edge case the heuristic passes miss — guaranteeing that any selection
    # the game engine itself considers valid will be accepted.
    if level_rank is not None:
        from guandan.combos import generate_all_leads
        sel_set = set(sel_cards)
        for gen_combo in generate_all_leads(sel_set, level_rank):
            if gen_combo.type == ComboType.PASS:
                continue
            if set(gen_combo.cards) != sel_set:
                continue  # must use ALL selected cards, not a subset
            for legal_combo in legal_moves:
                if legal_combo.type == gen_combo.type and legal_combo.key == gen_combo.key:
                    return [Combo(gen_combo.type, gen_combo.key, list(sel_cards),
                                  gen_combo.length, gen_combo.wild_count)]

    return []
