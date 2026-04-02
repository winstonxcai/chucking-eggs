"use client";

import { useMemo } from "react";
import type { ComboDTO } from "@/lib/types";

interface ComboBrowserProps {
  legalMoves: ComboDTO[];
  onSelectCombo: (combo: ComboDTO) => void;
}

const BOMB_TYPES = new Set([
  "BOMB_4", "BOMB_5", "STRAIGHT_FLUSH", "BOMB_6", "BOMB_7",
  "BOMB_8", "BOMB_9", "BOMB_10", "BOMB_JOKER",
]);

const TYPE_LABELS: Record<string, string> = {
  SINGLE: "Singles",
  PAIR: "Pairs",
  TRIPLE: "Triples",
  FULL_HOUSE: "Full Houses",
  STRAIGHT: "Straights",
  TUBE: "Tubes",
  PLATE: "Plates",
  BOMB_4: "4-Bombs",
  BOMB_5: "5-Bombs",
  STRAIGHT_FLUSH: "Straight Flushes",
  BOMB_6: "6-Bombs",
  BOMB_7: "7-Bombs",
  BOMB_8: "8-Bombs",
  BOMB_9: "9-Bombs",
  BOMB_10: "10-Bombs",
  BOMB_JOKER: "Rocket",
};

const TYPE_ORDER = [
  "SINGLE", "PAIR", "TRIPLE", "FULL_HOUSE", "STRAIGHT", "TUBE", "PLATE",
  "BOMB_4", "BOMB_5", "STRAIGHT_FLUSH", "BOMB_6", "BOMB_7",
  "BOMB_8", "BOMB_9", "BOMB_10", "BOMB_JOKER",
];

export default function ComboBrowser({ legalMoves, onSelectCombo }: ComboBrowserProps) {
  const grouped = useMemo(() => {
    const groups: Map<string, ComboDTO[]> = new Map();
    for (const combo of legalMoves) {
      if (combo.is_pass) continue;
      if (!groups.has(combo.type)) groups.set(combo.type, []);
      groups.get(combo.type)!.push(combo);
    }
    // Return groups in canonical TYPE_ORDER
    const ordered: Map<string, ComboDTO[]> = new Map();
    for (const type of TYPE_ORDER) {
      if (groups.has(type)) ordered.set(type, groups.get(type)!);
    }
    return ordered;
  }, [legalMoves]);

  if (grouped.size === 0) {
    return (
      <div className="text-sm text-text-secondary">No combos available</div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {Array.from(grouped.entries()).map(([type, combos]) => {
        const label = TYPE_LABELS[type] ?? type;
        const isBomb = BOMB_TYPES.has(type);
        return (
          <div key={type} className="flex flex-col gap-2">
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-text-secondary">{"\u25b8"}</span>
              <span
                className={`text-[13px] font-semibold ${
                  isBomb ? "text-accent" : "text-foreground"
                }`}
              >
                {label}
              </span>
              <span className="text-xs text-text-secondary">({combos.length})</span>
            </div>
            <div className="flex flex-wrap gap-1.5 pl-4">
              {combos.map((combo, i) => (
                <button
                  key={i}
                  className={`px-2.5 py-1 text-[13px] font-semibold rounded-md cursor-pointer transition-colors ${
                    isBomb
                      ? "bg-[#FFF8F5] border border-accent text-accent hover:bg-accent hover:text-white"
                      : "bg-background border border-border text-foreground hover:border-accent hover:text-accent"
                  }`}
                  onClick={() => onSelectCombo(combo)}
                >
                  {combo.display}
                </button>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
