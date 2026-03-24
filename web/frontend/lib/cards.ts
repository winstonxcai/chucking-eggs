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

/** Group cards by rank for stacked display */
export function groupByRank(cards: CardDTO[]): CardDTO[][] {
  const groups: Map<number, CardDTO[]> = new Map();
  for (const card of cards) {
    const key = card.rank;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(card);
  }
  return Array.from(groups.values());
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
