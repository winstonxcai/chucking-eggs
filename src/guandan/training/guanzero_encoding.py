"""GuanZero state/action encoding (arXiv:2402.13582).

108-dim card encoding, 1075-dim non-history state, [5,432] LSTM history,
108-dim action encoding, and per-action behavior flags.

Card index layout (8×15 matrix, 12 excluded cells = 108 valid):
  rows  = suit + deck*4   (0=♠d1, 1=♥d1, 2=♦d1, 3=♣d1, 4=♠d2, 5=♥d2, 6=♦d2, 7=♣d2)
  cols  = rank-2 for 2-A (0-12), 13=BJ, 14=RJ
  Jokers: BJ(rank=16, suit=0/SPADE) in rows {0,4} at col 13
          RJ(rank=17, suit=1/HEART) in rows {1,5} at col 14
  Excluded: all other (row, col>=13) combos (12 cells)
"""

from __future__ import annotations

import numpy as np

from ..cards import Card, ComboType, Rank
from ..game import GuanDanEnv

# ---------------------------------------------------------------------------
# 108-dim card index lookup table (built once at import time)
# ---------------------------------------------------------------------------

_CARD_TO_IDX: dict[tuple[int, int, int], int] = {}  # (rank, suit, deck) -> index

_idx = 0
for _deck in range(2):
    for _suit in range(4):
        _row = _suit + _deck * 4
        for _rank_val in range(2, 18):
            if _rank_val == 15:
                continue  # rank 15 doesn't exist
            # Determine column
            if _rank_val <= 14:
                _col = _rank_val - 2          # 2→0, ..., A(14)→12
            elif _rank_val == 16:
                _col = 13                      # BLACK_JOKER
            elif _rank_val == 17:
                _col = 14                      # RED_JOKER
            else:
                continue

            # Skip excluded joker cells
            if _col == 13 and _suit not in (0,):    # BJ only in SPADE rows (suit=0)
                continue
            if _col == 14 and _suit not in (1,):    # RJ only in HEART rows (suit=1)
                continue

            _CARD_TO_IDX[(_rank_val, _suit, _deck)] = _idx
            _idx += 1

assert len(_CARD_TO_IDX) == 108, f"Expected 108 cards, got {len(_CARD_TO_IDX)}"
assert set(_CARD_TO_IDX.values()) == set(range(108)), "Indices 0-107 not all used"


def card_to_108(rank: int, suit: int, deck: int) -> int:
    """Return the index [0, 107] for this physical card."""
    return _CARD_TO_IDX[(rank, suit, deck)]


def cards_to_108(cards: set[Card] | frozenset[Card] | list[Card]) -> np.ndarray:
    """Set/list of Card objects → 108-dim binary float32 vector."""
    vec = np.zeros(108, dtype=np.float32)
    for c in cards:
        vec[_CARD_TO_IDX[(c.rank, c.suit, c.deck)]] = 1.0
    return vec


def combo_to_108(combo: "Combo") -> np.ndarray:  # noqa: F821
    """A Combo (played action) → 108-dim binary vector of its cards.
    PASS combo → all-zeros vector.
    """
    if combo.type == ComboType.PASS:
        return np.zeros(108, dtype=np.float32)
    return cards_to_108(combo.cards)


# ---------------------------------------------------------------------------
# Behavior flags helpers
# ---------------------------------------------------------------------------

def _get_last_action(env: GuanDanEnv, player: int):
    """Return the most recent non-trivial combo played by player, or None."""
    for p, combo in reversed(env.move_history):
        if p == player:
            return combo
    return None


def _is_highest_rank(combo, level_rank: int) -> bool:
    """Return True if combo is a triple of the level-rank card (highest possible lead)."""
    if combo.type != ComboType.TRIPLE:
        return False
    return all(c.rank == level_rank for c in combo.cards)


def compute_behavior_flags(
    env: GuanDanEnv,
    player: int,
    action,
    legal_moves: list,
) -> np.ndarray:
    """Compute 9-dim behavior flag vector for a specific (player, action) pair.

    Layout: [cooperating(3), dwarfing(3), assisting(3)]
    Each 3-dim group: [not_applicable, doing_it, refusing]
    """
    flags = np.zeros(9, dtype=np.float32)
    partner = (player + 2) % 4
    opp_left = (player + 1) % 4
    opp_right = (player - 1) % 4

    is_pass = action.type == ComboType.PASS
    is_leading = env.current_trick is None

    # ── Cooperating ─────────────────────────────────────────────────────────
    # Conditions: teammate's combo is the current trick AND no opponent beat it
    # since then AND current player has a legal non-pass that beats it.
    can_coop = False
    if not is_leading and env.current_trick is not None:
        # Check if trick_winner is our partner
        if env.trick_winner == partner:
            # Check that current player has at least one move that beats the trick
            can_beat = any(
                m.type != ComboType.PASS for m in legal_moves
            )
            can_coop = can_beat

    if not can_coop:
        flags[0] = 1.0  # [1,0,0] — not applicable
    elif is_pass:
        flags[1] = 1.0  # [0,1,0] — cooperating (choosing to pass)
    else:
        flags[2] = 1.0  # [0,0,1] — refusing to cooperate

    # ── Dwarfing ─────────────────────────────────────────────────────────────
    # Conditions: leading AND has a legal combo > min opponent hand size
    opp_hand_sizes = [len(env.hands[opp_left]), len(env.hands[opp_right])]
    # Exclude already-out opponents from dwarfing consideration
    active_opp_sizes = [s for p, s in zip([opp_left, opp_right], opp_hand_sizes)
                        if not env.is_out[opp_left if p == opp_left else opp_right]]
    min_opp = min(active_opp_sizes) if active_opp_sizes else 27

    can_dwarf = is_leading and any(
        m.type != ComboType.PASS and len(m.cards) > min_opp
        for m in legal_moves
    )

    action_card_count = len(action.cards) if not is_pass else 0

    if not can_dwarf:
        flags[3] = 1.0
    elif not is_pass and action_card_count > min_opp:
        flags[4] = 1.0  # dwarfing
    else:
        flags[5] = 1.0  # refusing

    # ── Assisting ─────────────────────────────────────────────────────────────
    # Conditions: leading AND has combo < partner hand size AND not highest rank
    partner_hand_size = len(env.hands[partner])
    can_assist = is_leading and any(
        m.type != ComboType.PASS
        and len(m.cards) < partner_hand_size
        and not _is_highest_rank(m, env.level_rank)
        for m in legal_moves
    )

    if not can_assist:
        flags[6] = 1.0
    elif (
        not is_pass
        and action_card_count < partner_hand_size
        and not _is_highest_rank(action, env.level_rank)
    ):
        flags[7] = 1.0  # assisting
    else:
        flags[8] = 1.0  # refusing

    return flags


# ---------------------------------------------------------------------------
# Base state encoding (everything except behavior flags)
# ---------------------------------------------------------------------------

def encode_base_state(env: GuanDanEnv, player: int, level_rank: int) -> np.ndarray:
    """Encode the 1066-dim base state (all features except the 9 behavior flags).

    Features in order:
      1. own hand cards              108
      2. union of other 3 hands      108
      3. most recent action × 4      432   (seat-relative: self, left, partner, right)
      4. played cards × 3 others     324   (left, partner, right)
      5. remaining count one-hot × 3  81   (left, partner, right)
      6. level card one-hot           13
    Total: 1066
    """
    # Seat order relative to current player: [self, left_opp, partner, right_opp]
    # Counterclockwise play: left opponent = (player-1)%4, right = (player+1)%4
    seats = [player, (player - 1) % 4, (player + 2) % 4, (player + 1) % 4]
    other_seats = seats[1:]  # [left_opp, partner, right_opp]

    # 1. Own hand (108)
    own_hand = cards_to_108(env.hands[player])

    # 2. Union of all other hands (108) — oracle info, same as DanZero/DouZero
    other_cards: set[Card] = set()
    for p in other_seats:
        other_cards |= env.hands[p]
    others_union = cards_to_108(other_cards)

    # 3. Most recent action of each player (4 × 108 = 432), seat-relative order
    recent_actions = np.zeros(4 * 108, dtype=np.float32)
    for slot, p in enumerate(seats):
        last = _get_last_action(env, p)
        if last is not None and last.type != ComboType.PASS:
            recent_actions[slot * 108:(slot + 1) * 108] = combo_to_108(last)

    # 4. Played cards of each other player (3 × 108 = 324)
    played = np.zeros(3 * 108, dtype=np.float32)
    for slot, p in enumerate(other_seats):
        played[slot * 108:(slot + 1) * 108] = cards_to_108(env.played[p])

    # 5. Remaining card count one-hot (3 × 27 = 81)
    #    One-hot over {1, ..., 27}: index n-1 is hot when player has n cards.
    #    All zeros if player is out (0 cards).
    remaining = np.zeros(3 * 27, dtype=np.float32)
    for slot, p in enumerate(other_seats):
        n = len(env.hands[p])
        if 1 <= n <= 27:
            remaining[slot * 27 + (n - 1)] = 1.0

    # 6. Current level card one-hot (13 dims, rank 2→0, A(14)→12)
    level_onehot = np.zeros(13, dtype=np.float32)
    level_onehot[level_rank - 2] = 1.0

    base = np.concatenate([
        own_hand,        # 108
        others_union,    # 108
        recent_actions,  # 432
        played,          # 324
        remaining,       # 81
        level_onehot,    # 13
    ])
    assert base.shape == (1066,), f"Base state shape mismatch: {base.shape}"
    return base


# ---------------------------------------------------------------------------
# History encoding (LSTM input)
# ---------------------------------------------------------------------------

def encode_history(env: GuanDanEnv, player: int, max_rounds: int = 5) -> np.ndarray:
    """Build LSTM input: [max_rounds, 4*108] = [5, 432].

    Each row = one "round" = 4 consecutive actions, one per seat slot.
    Seat slots are relative to current player:
      slot 0 = current player (self)
      slot 1 = left opponent (player-1)%4
      slot 2 = partner (player+2)%4
      slot 3 = right opponent (player+1)%4

    PASS or no-action = all zeros. Padded with zeros at the front if
    fewer than max_rounds rounds of history exist. Oldest round first.
    """
    seat_to_slot = {
        player: 0,
        (player - 1) % 4: 1,
        (player + 2) % 4: 2,
        (player + 1) % 4: 3,
    }

    history = np.zeros((max_rounds, 4 * 108), dtype=np.float32)

    # Take last max_rounds*4 moves from move_history
    recent = env.move_history[-(max_rounds * 4):]

    # Group into rounds of 4: iterate oldest-first
    # Build list of rounds, each round = dict {slot: combo_vec}
    rounds: list[dict[int, np.ndarray]] = []
    current_round: dict[int, np.ndarray] = {}
    seats_seen: set[int] = set()

    for seat, combo in recent:
        slot = seat_to_slot.get(seat)
        if slot is None:
            continue

        # If we've already seen this slot in the current round, start a new round
        if slot in seats_seen:
            rounds.append(current_round)
            current_round = {}
            seats_seen = set()

        if combo.type != ComboType.PASS:
            current_round[slot] = combo_to_108(combo)
        seats_seen.add(slot)

    if current_round:
        rounds.append(current_round)

    # Fill history array: take last max_rounds rounds, oldest first
    rounds = rounds[-max_rounds:]
    offset = max_rounds - len(rounds)
    for i, rnd in enumerate(rounds):
        for slot, vec in rnd.items():
            history[offset + i, slot * 108:(slot + 1) * 108] = vec

    return history


def _count_valid_history_steps(env: GuanDanEnv, player: int, max_rounds: int = 5) -> int:
    """Return how many valid (non-zero) history steps are present, clamped to [1, max_rounds]."""
    n_moves = len(env.move_history)
    # Rough estimate: each full round = 4 moves
    full_rounds = min(max_rounds, n_moves // 4)
    return max(1, full_rounds)


# ---------------------------------------------------------------------------
# Full state encoding
# ---------------------------------------------------------------------------

def encode_guanzero_state(
    env: GuanDanEnv,
    player: int,
    action,
    legal_moves: list,
    level_rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode the full GuanZero state for a specific (player, action) pair.

    Returns:
        non_history:  np.ndarray [1075]   — base state + behavior flags
        history:      np.ndarray [5, 432] — LSTM input (chronological)
        action_enc:   np.ndarray [108]    — action encoding
    """
    base = encode_base_state(env, player, level_rank)          # [1066]
    behavior = compute_behavior_flags(env, player, action, legal_moves)  # [9]
    non_history = np.concatenate([base, behavior])             # [1075]

    history = encode_history(env, player)                      # [5, 432]
    action_enc = combo_to_108(action)                          # [108]

    assert non_history.shape == (1075,), f"Expected (1075,), got {non_history.shape}"
    assert history.shape == (5, 432), f"Expected (5, 432), got {history.shape}"
    assert action_enc.shape == (108,), f"Expected (108,), got {action_enc.shape}"

    return non_history, history, action_enc


# ---------------------------------------------------------------------------
# Batched action scoring (LSTM-once optimization)
# ---------------------------------------------------------------------------

def score_all_actions(
    net,
    env: GuanDanEnv,
    player: int,
    legal_moves: list,
    level_rank: int,
    device,
) -> "torch.Tensor":  # noqa: F821
    """Score all legal actions with one LSTM pass shared across all B actions.

    Uses the LSTM-once optimization: run LSTM once on history, then batch
    all B MLP forwards using the shared LSTM output.

    Returns: Tensor [B] of Q-values.
    """
    import torch

    B = len(legal_moves)
    if B == 0:
        return torch.zeros(0, device=device)

    # Compute base state once (shared across all actions)
    base = encode_base_state(env, player, level_rank)  # [1066]
    history = encode_history(env, player)              # [5, 432]
    hist_len = _count_valid_history_steps(env, player)

    non_histories = []
    actions = []
    for action in legal_moves:
        behavior = compute_behavior_flags(env, player, action, legal_moves)  # [9]
        nh = np.concatenate([base, behavior])          # [1075]
        non_histories.append(nh)
        actions.append(combo_to_108(action))           # [108]

    nh_batch = torch.tensor(np.array(non_histories), dtype=torch.float32, device=device)  # [B, 1075]
    hist_tensor = torch.tensor(history, dtype=torch.float32, device=device)               # [5, 432]
    act_batch = torch.tensor(np.array(actions), dtype=torch.float32, device=device)       # [B, 108]
    hl = torch.tensor([hist_len], dtype=torch.long, device=device)                        # [1]

    with torch.no_grad():
        q_values = net.forward_fast(nh_batch, hist_tensor, hl, act_batch)  # [B]

    return q_values
