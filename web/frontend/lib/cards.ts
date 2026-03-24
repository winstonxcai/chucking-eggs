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

/** Check if a set of selected card IDs matches any legal combo */
export function findMatchingCombo(
  selectedIds: Set<string>,
  legalMoves: ComboDTO[]
): ComboDTO | null {
  if (selectedIds.size === 0) return null;
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

  // Straight (5 consecutive, not all same suit)
  if (n === 5 && uniqueRanks.length === 5) {
    const isConsecutive = uniqueRanks[4] - uniqueRanks[0] === 4;
    // Ace-low: A-2-3-4-5
    const isAceLow = uniqueRanks[0] === 2 && uniqueRanks[4] === 14 &&
      uniqueRanks[1] === 3 && uniqueRanks[2] === 4 && uniqueRanks[3] === 5;
    if (isConsecutive || isAceLow) {
      const allSameSuit = new Set(suits).size === 1;
      if (allSameSuit) {
        return { valid: true, type: "STRAIGHT_FLUSH", name: `Straight Flush ${RANK_NAMES[uniqueRanks[0]] ?? ""}-${RANK_NAMES[uniqueRanks[4]] ?? ""}` };
      }
      return { valid: true, type: "STRAIGHT", name: `Straight ${RANK_NAMES[uniqueRanks[0]] ?? ""}-${RANK_NAMES[uniqueRanks[4]] ?? ""}` };
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

/** Find all 5-card straight flushes for a given suit from a set of cards.
 *  Wild cards can fill gaps in a run.
 *  Returns array of {label, cards} for each valid 5-card window. */
export function findStraightFlushes(
  cards: CardDTO[],
  suit: number
): { label: string; cards: CardDTO[] }[] {
  // Non-wild cards of this suit
  const suitCards = cards
    .filter((c) => c.suit === suit && c.rank >= 2 && c.rank <= 14 && !c.is_wild)
    .sort((a, b) => a.rank - b.rank);

  // Wild cards (any suit) — can fill any gap
  const wildCards = cards.filter((c) => c.is_wild);

  // Build rank → CardDTO map (first card per rank for this suit)
  const rankToCard: Map<number, CardDTO> = new Map();
  for (const c of suitCards) {
    if (!rankToCard.has(c.rank)) rankToCard.set(c.rank, c);
  }

  const results: { label: string; cards: CardDTO[] }[] = [];

  // Check every possible 5-rank window (2-6, 3-7, ..., 10-A)
  for (let start = 2; start <= 10; start++) {
    const windowCards: CardDTO[] = [];
    let wildsNeeded = 0;

    for (let r = start; r < start + 5; r++) {
      if (rankToCard.has(r)) {
        windowCards.push(rankToCard.get(r)!);
      } else {
        wildsNeeded++;
      }
    }

    if (wildsNeeded <= wildCards.length) {
      const selected = [...windowCards, ...wildCards.slice(0, wildsNeeded)];
      const highRank = start + 4;
      const label = `${RANK_NAMES[highRank] ?? highRank}-high SF`;
      results.push({ label, cards: selected });
    }
  }

  return results;
}
