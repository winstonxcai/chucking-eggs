"""Serialize GuanDanEnv state to JSON for the frontend."""

from __future__ import annotations


from guandan.cards import BOMB_TYPES, Card, ComboType, Rank, Suit, is_wild, level_order_key
from guandan.combos import Combo, generate_all_leads, generate_responses
from guandan.game import GuanDanEnv

SUIT_SYMBOLS = {Suit.SPADE: "\u2660", Suit.HEART: "\u2665", Suit.DIAMOND: "\u2666", Suit.CLUB: "\u2663"}
RANK_NAMES = {
    2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
    9: "9", 10: "10", 11: "J", 12: "Q", 13: "K", 14: "A",
    Rank.BLACK_JOKER: "BJ", Rank.RED_JOKER: "RJ",
}
COMBO_TYPE_NAMES = {
    ComboType.PASS: "Pass",
    ComboType.SINGLE: "Single",
    ComboType.PAIR: "Pair",
    ComboType.TRIPLE: "Triple",
    ComboType.FULL_HOUSE: "Full House",
    ComboType.STRAIGHT: "Straight",
    ComboType.TUBE: "Tube",
    ComboType.PLATE: "Plate",
    ComboType.BOMB_4: "Bomb",
    ComboType.BOMB_5: "Bomb",
    ComboType.STRAIGHT_FLUSH: "Straight Flush",
    ComboType.BOMB_6: "Bomb",
    ComboType.BOMB_7: "Bomb",
    ComboType.BOMB_8: "Bomb",
    ComboType.BOMB_9: "Bomb",
    ComboType.BOMB_10: "Bomb",
    ComboType.BOMB_JOKER: "Rocket",
}
COMBO_TYPE_CHINESE = {
    ComboType.PASS: "\u8fc7",
    ComboType.SINGLE: "\u5355\u724c",
    ComboType.PAIR: "\u5bf9\u5b50",
    ComboType.TRIPLE: "\u4e09\u6761",
    ComboType.FULL_HOUSE: "\u4e09\u5e26\u4e8c",
    ComboType.STRAIGHT: "\u987a\u5b50",
    ComboType.TUBE: "\u8fde\u5bf9",
    ComboType.PLATE: "\u94a2\u677f",
    ComboType.BOMB_4: "\u70b8\u5f39",
    ComboType.BOMB_5: "\u70b8\u5f39",
    ComboType.STRAIGHT_FLUSH: "\u540c\u82b1\u987a",
    ComboType.BOMB_6: "\u70b8\u5f39",
    ComboType.BOMB_7: "\u70b8\u5f39",
    ComboType.BOMB_8: "\u70b8\u5f39",
    ComboType.BOMB_9: "\u70b8\u5f39",
    ComboType.BOMB_10: "\u70b8\u5f39",
    ComboType.BOMB_JOKER: "\u5929\u738b\u70b8",
}


def card_to_dto(card: Card) -> dict:
    is_joker = card.rank in (Rank.BLACK_JOKER, Rank.RED_JOKER)
    rank_display = RANK_NAMES.get(card.rank, "?")
    suit_symbol = "" if is_joker else SUIT_SYMBOLS.get(card.suit, "?")
    return {
        "rank": card.rank,
        "suit": card.suit,
        "deck": card.deck,
        "id": f"{card.rank}-{card.suit}-{card.deck}",
        "display": rank_display if is_joker else f"{rank_display}{suit_symbol}",
        "rank_display": rank_display,
        "suit_symbol": suit_symbol,
    }


def combo_to_dto(combo: Combo, level_rank: int) -> dict:
    if combo.type == ComboType.PASS:
        return {"type": "PASS", "type_id": 0, "type_name": "Pass",
                "type_chinese": "\u8fc7", "cards": [], "is_pass": True}
    cards = [card_to_dto(c) for c in combo.cards]
    display_parts = [c["display"] for c in cards]
    return {
        "type": combo.type.name,
        "type_id": int(combo.type),
        "type_name": COMBO_TYPE_NAMES.get(combo.type, "Unknown"),
        "type_chinese": COMBO_TYPE_CHINESE.get(combo.type, ""),
        "cards": cards,
        "display": " ".join(display_parts),
        "key": combo.key,
        "wild_count": combo.wild_count,
        "is_pass": False,
    }


def combo_sort_key(combo: Combo, level_rank: int) -> tuple:
    """Sort key for ordering combos smallest → biggest."""
    is_bomb = combo.type in BOMB_TYPES
    if is_bomb:
        bomb_tier = int(combo.type)
        if combo.type == ComboType.STRAIGHT_FLUSH:
            rank_key = combo.key  # natural order
        elif combo.type == ComboType.BOMB_JOKER:
            rank_key = 99
        else:
            rank_key = level_order_key(combo.key, level_rank)
        return (1, bomb_tier, rank_key, 0)
    elif combo.type == ComboType.FULL_HOUSE:
        triple_key = level_order_key(combo.key, level_rank)
        # Pair rank: any non-wild card whose rank isn't the triple rank
        non_triple_natural = [
            c.rank for c in combo.cards
            if not is_wild(c, level_rank) and c.rank != combo.key
        ]
        pair_rank = non_triple_natural[0] if non_triple_natural else level_rank
        pair_key = level_order_key(pair_rank, level_rank)
        return (0, int(combo.type), triple_key, pair_key)
    else:
        if combo.type in (ComboType.STRAIGHT, ComboType.TUBE, ComboType.PLATE):
            key_val = combo.key  # natural order
        else:
            key_val = level_order_key(combo.key, level_rank)
        return (0, int(combo.type), key_val, 0)


def sort_hand(cards: set[Card], level_rank: int) -> list[Card]:
    """Sort hand by level order, then suit, then deck."""
    def sort_key(c: Card):
        if c.rank in (Rank.BLACK_JOKER, Rank.RED_JOKER):
            return (100 + c.rank, c.suit, c.deck)
        if is_wild(c, level_rank):
            return (99, c.suit, c.deck)
        return (level_order_key(c.rank, level_rank), c.suit, c.deck)
    return sorted(cards, key=sort_key)



def _compute_sf_options(
    hand: set[Card],
    grouped_ids: set[str],
    level_rank: int,
) -> dict[str, list[dict]]:
    """Compute SF options by suit from non-grouped hand cards.

    Returns {suit_str: [{label, cardIds}]} where suit_str is "0"-"3".
    Deduplicates by natural-card set so each unique SF appears only once.
    """
    ungrouped = {c for c in hand if f"{c.rank}-{c.suit}-{c.deck}" not in grouped_ids}
    sf_combos = [
        combo for combo in generate_all_leads(ungrouped, level_rank)
        if combo.type == ComboType.STRAIGHT_FLUSH
    ]

    options: dict[str, list[dict]] = {"0": [], "1": [], "2": [], "3": []}
    seen: set[frozenset] = set()
    for combo in sf_combos:
        card_ids = [f"{c.rank}-{c.suit}-{c.deck}" for c in combo.cards]
        key = frozenset(card_ids)
        if key in seen:
            continue
        seen.add(key)
        naturals = [c for c in combo.cards if not is_wild(c, level_rank)]
        if not naturals:
            continue
        suit = naturals[0].suit
        top = combo.key  # top rank of window: 5 = A-low, 14 = A-high
        label = "A-5 SF" if top == 5 else f"{RANK_NAMES.get(top, top)}-high SF"
        options[str(suit)].append({"label": label, "cardIds": card_ids})
    return options


def _rotate(seat: int, human_seat: int) -> int:
    """Rotate absolute seat to viewer-relative seat (viewer is always 0)."""
    return (seat - human_seat + 4) % 4


def serialize_game_state(
    env: GuanDanEnv,
    game_id: str,
    human_seat: int,
    player_infos: list[dict],
    trick_plays: dict[int, Combo] | None = None,
    groups: list[dict] | None = None,
    mode: str = "solo",
) -> dict:
    """Serialize full game state from the human player's perspective."""
    sorted_hand = sort_hand(env.hands[human_seat], env.level_rank)
    hand_dtos = [card_to_dto(c) for c in sorted_hand]

    # Mark wild cards
    for dto, card in zip(hand_dtos, sorted_hand):
        dto["is_wild"] = is_wild(card, env.level_rank)

    # Player infos with card counts — seats rotated so viewer is always seat 0
    players = []
    for i, info in enumerate(player_infos):
        card_count = len(env.hands[i])
        players.append({
            "seat": _rotate(i, human_seat),
            "name": info["name"],
            "avatar": info.get("avatar"),
            "elo": info.get("elo", 1200),
            "card_count": card_count,
            "is_out": env.is_out[i],
            "is_teammate": (i % 2) == (human_seat % 2),
            "is_human": i == human_seat,
        })

    # Per-seat trick actions (geometric layout) — keys rotated to viewer-relative seats
    trick_actions: dict[str, dict | None] = {"0": None, "1": None, "2": None, "3": None}
    if trick_plays:
        for seat, combo in trick_plays.items():
            rotated_key = str(_rotate(seat, human_seat))
            if combo.type == ComboType.PASS:
                trick_actions[rotated_key] = {"type": "pass"}
            else:
                trick_actions[rotated_key] = {
                    "type": "play",
                    "combo": combo_to_dto(combo, env.level_rank),
                }

    # Legal moves sorted smallest → biggest (only when it's the human's turn)
    legal_moves = []
    is_my_turn = env.current_player == human_seat and not env.done
    grouped_ids = {cid for g in (groups or []) for cid in g["cardIds"]}
    ungrouped_hand = {
        c for c in env.hands[human_seat]
        if f"{c.rank}-{c.suit}-{c.deck}" not in grouped_ids
    }
    if is_my_turn:
        if env.current_trick is None:
            raw_moves = generate_all_leads(ungrouped_hand, env.level_rank)
        else:
            raw_moves = generate_responses(ungrouped_hand, env.level_rank, env.current_trick)
        non_pass = [c for c in raw_moves if c.type != ComboType.PASS]
        non_pass.sort(key=lambda c: combo_sort_key(c, env.level_rank))
        for combo in non_pass:
            legal_moves.append(combo_to_dto(combo, env.level_rank))
        if any(c.type == ComboType.PASS for c in raw_moves):
            legal_moves.append(combo_to_dto(
                next(c for c in raw_moves if c.type == ComboType.PASS),
                env.level_rank,
            ))
    if is_my_turn and groups:
        hand_by_id = {f"{c.rank}-{c.suit}-{c.deck}": c for c in env.hands[human_seat]}
        for group in groups:
            group_cards = {hand_by_id[cid] for cid in group["cardIds"] if cid in hand_by_id}
            if not group_cards:
                continue
            if env.current_trick is None:
                grp_moves = generate_all_leads(group_cards, env.level_rank)
            else:
                grp_moves = generate_responses(group_cards, env.level_rank, env.current_trick)
            grp_non_pass = [c for c in grp_moves if c.type != ComboType.PASS]
            grp_non_pass.sort(key=lambda c: combo_sort_key(c, env.level_rank))
            insert_pos = len(legal_moves)
            if legal_moves and legal_moves[-1].get("is_pass"):
                insert_pos -= 1
            for combo in grp_non_pass:
                legal_moves.insert(insert_pos, combo_to_dto(combo, env.level_rank))
                insert_pos += 1

    # All combos from full hand in leading context (for hand analysis sidebar)
    all_lead_raw = generate_all_leads(env.hands[human_seat], env.level_rank)
    all_lead = [c for c in all_lead_raw if c.type != ComboType.PASS]
    all_lead.sort(key=lambda c: combo_sort_key(c, env.level_rank))
    all_moves = [combo_to_dto(c, env.level_rank) for c in all_lead]

    sf_options = _compute_sf_options(env.hands[human_seat], grouped_ids, env.level_rank)

    # Reveal partner's hand once the human player has finished
    partner_seat = (human_seat + 2) % 4
    if human_seat in env.finish_order and env.hands[partner_seat]:
        p_sorted = sort_hand(env.hands[partner_seat], env.level_rank)
        partner_hand = [card_to_dto(c) for c in p_sorted]
    else:
        partner_hand = None

    # Reveal opponent hands once the human player has finished
    opp_seats = [(human_seat + 1) % 4, (human_seat + 3) % 4]
    opponent_hands: dict[str, list[dict]] = {}
    if human_seat in env.finish_order:
        for opp_seat in opp_seats:
            if env.hands[opp_seat]:
                rotated = str(_rotate(opp_seat, human_seat))
                p_sorted = sort_hand(env.hands[opp_seat], env.level_rank)
                opponent_hands[rotated] = [card_to_dto(c) for c in p_sorted]

    trick_lead_seat = _rotate(env.trick_winner, human_seat) if env.trick_winner is not None else None

    return {
        "game_id": game_id,
        "my_seat": 0,
        "current_player": _rotate(env.current_player, human_seat),
        "is_my_turn": is_my_turn,
        "my_hand": hand_dtos,
        "players": players,
        "trick_actions": trick_actions,
        "is_leading": env.is_leading(),
        "legal_moves": legal_moves,
        "finish_order": [_rotate(s, human_seat) for s in env.finish_order],
        "done": env.done,
        "level_rank": env.level_rank,
        "rewards": env.get_rewards() if env.done else None,
        "groups": groups or [],
        "sf_options": sf_options,
        "partner_hand": partner_hand,
        "opponent_hands": opponent_hands,
        "all_moves": all_moves,
        "trick_lead_seat": trick_lead_seat,
        "mode": mode,
    }
