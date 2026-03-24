"use client";

import { useMemo } from "react";
import type { CardDTO } from "@/lib/types";
import { groupByRank } from "@/lib/cards";
import CardComponent from "./CardComponent";

interface PlayerHandProps {
  cards: CardDTO[];
  selectedIds: Set<string>;
  onToggleCard: (id: string) => void;
}

export default function PlayerHand({ cards, selectedIds, onToggleCard }: PlayerHandProps) {
  const groups = useMemo(() => groupByRank(cards), [cards]);

  return (
    <div className="flex items-end justify-center gap-1.5">
      {groups.map((group) => {
        const anySelected = group.some((c) => selectedIds.has(c.id));
        return (
          <div
            key={group[0].rank + "-" + group[0].suit}
            className={`flex flex-col items-center transition-transform ${
              anySelected ? "-translate-y-3" : ""
            }`}
          >
            {/* Top card (full) */}
            <CardComponent
              card={group[0]}
              selected={selectedIds.has(group[0].id)}
              onClick={() => onToggleCard(group[0].id)}
            />
            {/* Additional cards: suit chips */}
            {group.slice(1).map((card) => (
              <div
                key={card.id}
                className={`w-9 h-[22px] flex items-center justify-center rounded cursor-pointer -mt-1 ${
                  selectedIds.has(card.id)
                    ? "bg-surface border-2 border-accent shadow-[0_1px_4px_rgba(217,119,87,0.15)]"
                    : "bg-surface border-[1.5px] border-border shadow-[0_1px_2px_rgba(0,0,0,0.06)]"
                }`}
                onClick={() => onToggleCard(card.id)}
              >
                <span
                  className="text-[13px] font-semibold"
                  style={{
                    color:
                      card.suit === 1 || card.suit === 2
                        ? "#C75D4A"
                        : "#1A1612",
                  }}
                >
                  {card.suit_symbol}
                </span>
              </div>
            ))}
          </div>
        );
      })}
    </div>
  );
}
