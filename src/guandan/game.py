"""4-player Guan Dan game environment.

Counterclockwise play, strict auto-pass logic (out players count as passing),
partner-leads rule, team-based game end.
"""

from __future__ import annotations

import random

from .cards import Card, ComboType, Rank, make_deck
from .combos import Combo, generate_all_leads, generate_responses


class GuanDanEnv:
    def __init__(self, level_rank: int = Rank.TWO):
        self.level_rank = level_rank
        self.reset()

    def reset(self) -> None:
        deck = make_deck()
        random.shuffle(deck)
        self.hands: list[set[Card]] = [
            set(deck[0:27]),
            set(deck[27:54]),
            set(deck[54:81]),
            set(deck[81:108]),
        ]
        self.played: list[set[Card]] = [set() for _ in range(4)]
        self.current_player: int = random.randint(0, 3)
        self.current_trick: Combo | None = None
        self.trick_winner: int | None = None
        self.consecutive_passes: int = 0  # counts all passes since last play (incl. auto)
        self.finish_order: list[int] = []
        self.is_out: list[bool] = [False] * 4
        self.done: bool = False
        self.move_history: list[tuple[int, Combo]] = []

    # ─── Seat helpers ───────────────────────────────────

    @staticmethod
    def partner(player: int) -> int:
        """Partner seat: 0↔2, 1↔3."""
        return (player + 2) % 4

    @staticmethod
    def _next_seat(player: int) -> int:
        """Next seat counterclockwise (subtract 1 mod 4)."""
        return (player - 1) % 4

    def _next_active_player(self, from_player: int) -> int | None:
        """Next active (non-out) player counterclockwise."""
        p = self._next_seat(from_player)
        for _ in range(4):
            if not self.is_out[p]:
                return p
            p = self._next_seat(p)
        return None

    def _active_count(self) -> int:
        return sum(1 for x in self.is_out if not x)

    # ─── Game end logic ─────────────────────────────────

    def _winning_team_done(self) -> bool:
        """Game ends once both players on a team have gone out."""
        return (self.is_out[0] and self.is_out[2]) or (self.is_out[1] and self.is_out[3])

    def _finalize_game(self) -> None:
        """Add remaining players to finish order and end game."""
        for p in range(4):
            if p not in self.finish_order:
                self.finish_order.append(p)
        self.done = True

    # ─── Trick resolution ───────────────────────────────

    def _end_trick(self) -> None:
        """Reset trick state for a new trick."""
        self.current_trick = None

    def _resolve_leader(self) -> int:
        """Who leads the next trick.

        Normally the trick winner. If they're out, their partner leads.
        If partner is also out, next active player counterclockwise.
        """
        assert self.trick_winner is not None
        leader = self.trick_winner
        if not self.is_out[leader]:
            return leader
        partner = self.partner(leader)
        if not self.is_out[partner]:
            return partner
        # Both out — find next active player
        result = self._next_active_player(leader)
        assert result is not None
        return result

    def _count_auto_passes_between(self, from_seat: int, to_seat: int) -> int:
        """Count how many out players sit between from_seat and to_seat (exclusive)
        going counterclockwise. These count as auto-passes."""
        count = 0
        seat = self._next_seat(from_seat)
        while seat != to_seat:
            if self.is_out[seat]:
                count += 1
            seat = self._next_seat(seat)
        return count

    # ─── Main step ──────────────────────────────────────

    def step(self, combo: Combo) -> tuple[int, bool]:
        """Execute a play. Returns (next_current_player, done)."""
        player = self.current_player
        self.move_history.append((player, combo))

        if combo.type == ComboType.PASS:
            return self._handle_pass(player)

        return self._handle_play(player, combo)

    def _handle_pass(self, player: int) -> tuple[int, bool]:
        """Handle a pass action. Counts this pass plus any auto-passes from
        out players between this player and the next active player."""
        assert self.trick_winner is not None

        # This player passes: +1
        self.consecutive_passes += 1

        # Find next active player counterclockwise
        next_p = self._next_active_player(player)

        # Count auto-passes from out players between current and next active
        if next_p is not None:
            auto = self._count_auto_passes_between(player, next_p)
            self.consecutive_passes += auto

        # Trick ends if 3+ consecutive passes since last play,
        # or if no active players remain, or if we'd return to trick winner
        if self.consecutive_passes >= 3 or next_p is None:
            self._end_trick()
            if next_p is None:
                self._finalize_game()
            else:
                self.current_player = self._resolve_leader()
            return self.current_player, self.done

        # If next active player is the trick winner (and they haven't gone out),
        # the trick also ends (everyone else has passed)
        if next_p == self.trick_winner:
            self._end_trick()
            self.current_player = self._resolve_leader()
            return self.current_player, self.done

        self.current_player = next_p
        return self.current_player, self.done

    def _handle_play(self, player: int, combo: Combo) -> tuple[int, bool]:
        """Handle a non-pass play."""
        # Remove played cards from hand
        for card in combo.cards:
            self.hands[player].discard(card)
            self.played[player].add(card)

        self.current_trick = combo
        self.trick_winner = player
        self.consecutive_passes = 0

        # Check if player goes out
        if len(self.hands[player]) == 0:
            self.finish_order.append(player)
            self.is_out[player] = True

            if self._winning_team_done():
                self._finalize_game()
                return self.current_player, True

        # Find next active player, counting auto-passes from out players in between
        next_p = self._next_active_player(player)
        if next_p is None:
            self._finalize_game()
            return self.current_player, True

        # Count auto-passes from out players between the player who played and next_p
        auto = self._count_auto_passes_between(player, next_p)
        self.consecutive_passes = auto

        # If auto-passes already reach 3 (e.g., player played and went out,
        # and the next 3 seats are all out), trick ends
        if self.consecutive_passes >= 3:
            self._end_trick()
            self.current_player = self._resolve_leader()
            return self.current_player, self.done

        self.current_player = next_p
        return self.current_player, self.done

    # ─── Query methods ──────────────────────────────────

    def legal_moves(self, player: int | None = None) -> list[Combo]:
        if player is None:
            player = self.current_player
        if self.current_trick is None:
            return generate_all_leads(self.hands[player], self.level_rank)
        else:
            return generate_responses(
                self.hands[player], self.level_rank, self.current_trick
            )

    def get_rewards(self) -> dict[int, float]:
        """Team-level rewards: each player gets their team's level change."""
        fo = self.finish_order
        team_pos = tuple(sorted([fo.index(0), fo.index(2)]))
        LEVEL_CHANGE = {
            (0, 1): 3.0, (0, 2): 2.0, (0, 3): 1.0,
            (1, 2): -1.0, (1, 3): -2.0, (2, 3): -3.0,
        }
        r = LEVEL_CHANGE[team_pos]
        return {0: r, 2: r, 1: -r, 3: -r}

    def is_leading(self) -> bool:
        return self.current_trick is None
