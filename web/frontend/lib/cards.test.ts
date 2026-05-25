import { describe, expect, it } from "vitest";

import { findMatchingCombo, groupByRank, validateCombo } from "./cards";
import type { CardDTO, ComboDTO } from "./types";

function card(rank: number, suit: number, deck = 0, isWild = false): CardDTO {
  return {
    rank,
    suit,
    deck,
    id: `${rank}-${suit}-${deck}`,
    display: `${rank}`,
    rank_display: `${rank}`,
    suit_symbol: `${suit}`,
    is_wild: isWild,
  };
}

function combo(type: string, cards: CardDTO[], key?: number): ComboDTO {
  return {
    type,
    type_id: 0,
    type_name: type,
    type_chinese: type,
    cards,
    key,
    is_pass: false,
  };
}

describe("groupByRank", () => {
  it("groups cards by rank and sorts each rank by suit", () => {
    const grouped = groupByRank([
      card(5, 3),
      card(3, 2),
      card(5, 0),
      card(3, 1),
    ]);

    expect(grouped.map((group) => group.map((c) => c.id))).toEqual([
      ["5-0-0", "5-3-0"],
      ["3-1-0", "3-2-0"],
    ]);
  });
});

describe("findMatchingCombo", () => {
  it("matches exact selected card ids", () => {
    const c1 = card(9, 0);
    const c2 = card(9, 1);
    const legal = combo("PAIR", [c1, c2]);

    expect(findMatchingCombo(new Set([c1.id, c2.id]), [legal])).toBe(legal);
  });

  it("substitutes selected double-deck copies for deck-normalized matches", () => {
    const legal = combo("PAIR", [card(10, 0, 0), card(10, 1, 0)]);
    const selected = [card(10, 0, 1), card(10, 1, 1)];

    const match = findMatchingCombo(new Set(selected.map((c) => c.id)), [legal], selected);

    expect(match?.cards.map((c) => c.id)).toEqual(["10-0-1", "10-1-1"]);
  });

  it("matches n-of-a-kind bombs by rank and size", () => {
    const selected = [0, 1, 2, 3].map((suit) => card(7, suit));
    const legal = combo("BOMB_4", selected, 7);

    expect(findMatchingCombo(new Set(selected.map((c) => c.id)), [legal], selected)).toBe(legal);
  });
});

describe("validateCombo", () => {
  it("recognizes full houses", () => {
    expect(validateCombo([card(8, 0), card(8, 1), card(8, 2), card(6, 0), card(6, 1)])).toMatchObject({
      type: "FULL_HOUSE",
    });
  });

  it("recognizes straight flushes with a wild card", () => {
    expect(validateCombo([card(10, 0), card(11, 0), card(12, 0), card(13, 0), card(2, 0, 0, true)])).toMatchObject({
      type: "STRAIGHT_FLUSH",
    });
  });
});
