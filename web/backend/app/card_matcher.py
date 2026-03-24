"""Match selected card IDs to a legal Combo."""

from __future__ import annotations

from guandan.combos import Combo


def find_matching_combo(
    selected_card_ids: list[str],
    legal_moves: list[Combo],
) -> list[Combo]:
    """Find legal combos whose card set matches the selected cards.

    Returns a list of matching combos (usually 0 or 1, rarely >1).
    """
    target = frozenset(selected_card_ids)
    matches = []
    for combo in legal_moves:
        combo_ids = frozenset(
            f"{c.rank}-{c.suit}-{c.deck}" for c in combo.cards
        )
        if combo_ids == target:
            matches.append(combo)
    return matches
