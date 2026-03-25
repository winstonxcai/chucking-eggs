import { CardDTO, ComboDTO } from "./types";

export const SUIT_COLORS: Record<number, string> = {
  0: "text-foreground",   // spade - black
  1: "text-team-red",     // heart - red
  2: "text-team-red",     // diamond - red
  3: "text-foreground",   // club - black
};

export const COMBO_TYPE_DISPLAY: Record<string, string> = {
  PASS: "Pass",
  SINGLE: "Single",
  PAIR: "Pair",
  TRIPLE: "Triple",
  FULL_HOUSE: "Full House",
  STRAIGHT: "Straight",
  TUBE: "Tube",
  PLATE: "Plate",
  BOMB_4: "Bomb",
  BOMB_5: "Bomb",
  STRAIGHT_FLUSH: "Straight Flush",
  BOMB_6: "Bomb",
  BOMB_7: "Bomb",
  BOMB_8: "Bomb",
  BOMB_9: "Bomb",
  BOMB_10: "Bomb",
  BOMB_JOKER: "Rocket",
};

const RANK_NAMES: Record<number, string> = {
  2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
  9: "9", 10: "10", 11: "J", 12: "Q", 13: "K", 14: "A",
  16: "BJ", 17: "RJ",
};

/** Group cards by rank for stacked display, sorted by suit within each group (♠♥♦♣) */
export function groupByRank(cards: CardDTO[]): CardDTO[][] {
  const groups: Map<number, CardDTO[]> = new Map();
  for (const card of cards) {
    const key = card.rank;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(card);
  }
  return Array.from(groups.values()).map((g) =>
    g.slice().sort((a, b) => a.suit - b.suit)
  );
}

/** Check if a set of selected card IDs matches any legal combo.
 *
 *  Pass 1: exact ID match (rank-suit-deck).
 *  Pass 2: deck-normalized — same (rank,suit) multiset, different deck.
 *          Returns the legal combo with the user's selected card objects
 *          substituted in, so the correct deck copies are played.
 *  Pass 3: N-of-a-kind bomb — all same rank, matching bomb type in legal moves.
 */
export function findMatchingCombo(
  selectedIds: Set<string>,
  legalMoves: ComboDTO[],
  hand?: CardDTO[]
): ComboDTO | null {
  if (selectedIds.size === 0) return null;

  // Pass 1: exact ID match
  for (const combo of legalMoves) {
    if (combo.is_pass) continue;
    const comboIds = new Set(combo.cards.map((c) => c.id));
    if (
      comboIds.size === selectedIds.size &&
      [...comboIds].every((id) => selectedIds.has(id))
    ) {
      return combo;
    }
  }

  // Resolve selected card objects for Pass 2 substitution
  const selectedCards = hand
    ? Array.from(selectedIds)
        .map((id) => hand.find((c) => c.id === id))
        .filter((c): c is CardDTO => c !== undefined)
    : [];

  // Pass 2: deck-normalized — compare (rank,suit) multisets ignoring deck
  const selRS = [...selectedIds]
    .map((id) => id.split("-").slice(0, 2).join("-"))
    .sort()
    .join(",");
  for (const combo of legalMoves) {
    if (combo.is_pass || combo.cards.length !== selectedIds.size) continue;
    const comboRS = combo.cards
      .map((c) => `${c.rank}-${c.suit}`)
      .sort()
      .join(",");
    if (comboRS === selRS) {
      // Return with user's actual cards so the right deck copies get played
      return selectedCards.length === combo.cards.length
        ? { ...combo, cards: selectedCards }
        : combo;
    }
  }

  // Pass 3: N-of-a-kind bomb — all selected cards share the same rank
  const selRanks = new Set([...selectedIds].map((id) => id.split("-")[0]));
  if (selRanks.size === 1) {
    const rank = Number([...selRanks][0]);
    const n = selectedIds.size;
    const bombType = n >= 4 ? `BOMB_${n}` : null;
    if (bombType) {
      for (const combo of legalMoves) {
        if (!combo.is_pass && combo.type === bombType && combo.key === rank) {
          return combo;
        }
      }
    }
  }

  return null;
}

/** Validate if a set of cards forms a recognized combo type (client-side). */
export function validateCombo(
  cards: CardDTO[]
): { valid: boolean; type: string; name: string } | null {
  if (cards.length === 0) return null;

  const ranks = cards.map((c) => c.rank).sort((a, b) => a - b);
  const suits = cards.map((c) => c.suit);
  const rankCounts: Map<number, number> = new Map();
  for (const r of ranks) {
    rankCounts.set(r, (rankCounts.get(r) || 0) + 1);
  }
  const uniqueRanks = [...rankCounts.keys()].sort((a, b) => a - b);
  const counts = [...rankCounts.values()].sort((a, b) => b - a);

  const n = cards.length;

  // Single
  if (n === 1) return { valid: true, type: "SINGLE", name: `${RANK_NAMES[ranks[0]] ?? "?"} Single` };

  // Pair
  if (n === 2 && uniqueRanks.length === 1)
    return { valid: true, type: "PAIR", name: `Pair of ${RANK_NAMES[ranks[0]] ?? "?"}s` };

  // Triple
  if (n === 3 && uniqueRanks.length === 1)
    return { valid: true, type: "TRIPLE", name: `Triple ${RANK_NAMES[ranks[0]] ?? "?"}s` };

  // Bomb (4+ of a kind)
  if (uniqueRanks.length === 1 && n >= 4)
    return { valid: true, type: `BOMB_${n}`, name: `Bomb ${RANK_NAMES[ranks[0]] ?? "?"} x${n}` };

  // Full House (3+2)
  if (n === 5 && counts[0] === 3 && counts[1] === 2) {
    const tripleRank = [...rankCounts.entries()].find(([, c]) => c === 3)?.[0];
    return { valid: true, type: "FULL_HOUSE", name: `Full House ${RANK_NAMES[tripleRank ?? 0] ?? "?"}` };
  }

  // Straight / Straight Flush (wild-card-aware)
  if (n === 5) {
    const nonWild = cards.filter((c) => !c.is_wild);
    const wildCount = n - nonWild.length;
    const nwRanks = nonWild.map((c) => c.rank).sort((a, b) => a - b);
    const uniqueNWRanks = [...new Set(nwRanks)];

    if (uniqueNWRanks.length === nonWild.length) {  // no duplicate non-wild ranks
      const span = uniqueNWRanks.length > 1
        ? uniqueNWRanks[uniqueNWRanks.length - 1] - uniqueNWRanks[0]
        : 0;
      if (span <= 4) {
        const gaps = span > 0 ? span - (uniqueNWRanks.length - 1) : 0;
        if (gaps <= wildCount) {
          const allSameSuit = nonWild.length === 0 ||
            new Set(nonWild.map((c) => c.suit)).size === 1;
          const lo = uniqueNWRanks[0] ?? 0;
          const hi = uniqueNWRanks[uniqueNWRanks.length - 1] ?? 0;
          // Wilds fill highest positions first → extend window upward
          const extra = wildCount - gaps;
          const fullHi = Math.min(14, hi + extra);
          const fullLo = lo - Math.max(0, hi + extra - 14);
          if (allSameSuit) {
            return { valid: true, type: "STRAIGHT_FLUSH", name: `Straight Flush ${RANK_NAMES[fullLo] ?? fullLo}-${RANK_NAMES[fullHi] ?? fullHi}` };
          }
          return { valid: true, type: "STRAIGHT", name: `Straight ${RANK_NAMES[fullLo] ?? fullLo}-${RANK_NAMES[fullHi] ?? fullHi}` };
        }
      }

      // Ace-low fallback: remap rank 14 → 1 and retry (handles A-2-3-4-5 window)
      if (uniqueNWRanks.includes(14)) {
        const lowRanks = uniqueNWRanks.map(r => r === 14 ? 1 : r).sort((a, b) => a - b);
        const lowSpan = lowRanks[lowRanks.length - 1] - lowRanks[0];
        if (lowSpan <= 4) {
          const lowGaps = lowSpan > 0 ? lowSpan - (lowRanks.length - 1) : 0;
          if (lowGaps <= wildCount) {
            const allSameSuit = nonWild.length === 0 ||
              new Set(nonWild.map((c) => c.suit)).size === 1;
            const hi = lowRanks[lowRanks.length - 1];
            const fullHi = hi + (wildCount - lowGaps);
            if (allSameSuit) {
              return { valid: true, type: "STRAIGHT_FLUSH", name: `Straight Flush A-${RANK_NAMES[fullHi] ?? fullHi}` };
            }
            return { valid: true, type: "STRAIGHT", name: `Straight A-${RANK_NAMES[fullHi] ?? fullHi}` };
          }
        }
      }
    }
  }

  // Tube (3 consecutive pairs, 6 cards)
  if (n === 6 && uniqueRanks.length === 3 && counts.every((c) => c === 2)) {
    const isConsecutive = uniqueRanks[2] - uniqueRanks[0] === 2;
    if (isConsecutive) {
      return { valid: true, type: "TUBE", name: `Tube ${RANK_NAMES[uniqueRanks[0]] ?? ""}-${RANK_NAMES[uniqueRanks[2]] ?? ""}` };
    }
  }

  // Plate (2 consecutive triples, 6 cards)
  if (n === 6 && uniqueRanks.length === 2 && counts.every((c) => c === 3)) {
    const isConsecutive = uniqueRanks[1] - uniqueRanks[0] === 1;
    if (isConsecutive) {
      return { valid: true, type: "PLATE", name: `Plate ${RANK_NAMES[uniqueRanks[0]] ?? ""}-${RANK_NAMES[uniqueRanks[1]] ?? ""}` };
    }
  }

  // Rocket (2BJ + 2RJ)
  if (n === 4 && ranks.filter((r) => r === 16).length === 2 && ranks.filter((r) => r === 17).length === 2) {
    return { valid: true, type: "BOMB_JOKER", name: "Rocket" };
  }

  return null;
}


