"use client";

import { useMemo } from "react";
import type { CardDTO, CardGroup } from "@/lib/types";
import { groupByRank } from "@/lib/cards";
import CardComponent from "./CardComponent";

interface PlayerHandProps {
  cards: CardDTO[];
  selectedIds: Set<string>;
  onToggleCard: (id: string) => void;
  groups: CardGroup[];
  groupedCardIds: Set<string>;
  onGroupClick: (group: CardGroup) => void;
  hiddenIds?: Set<string>;
  compact?: boolean;
}

export default function PlayerHand({
  cards,
  selectedIds,
  onToggleCard,
  groups,
  groupedCardIds,
  onGroupClick,
  hiddenIds,
}: PlayerHandProps) {
  // Ungrouped cards (not in any group, not flying)
  const ungroupedCards = useMemo(
    () => cards.filter((c) => !groupedCardIds.has(c.id) && !hiddenIds?.has(c.id)),
    [cards, groupedCardIds, hiddenIds]
  );
  const ungroupedGroups = useMemo(() => groupByRank(ungroupedCards), [ungroupedCards]);

  return (
    <div className="flex items-end justify-center gap-3">
      {/* Groups on the left */}
      {groups.map((group) => {
        // Hide group if any of its cards are flying
        if (hiddenIds && group.cardIds.some((cid) => hiddenIds.has(cid))) return null;
        const groupCards = cards.filter((c) => group.cardIds.includes(c.id));
        const isSelected = group.cardIds.some((cid) => selectedIds.has(cid));
        return (
          <div
            key={group.id}
            className={`flex flex-col items-center gap-1 p-2 rounded-lg cursor-pointer border transition-colors ${
              isSelected
                ? "border-accent bg-[#FFF8F5]"
                : "border-border bg-surface hover:border-accent/50"
            }`}
            onClick={() => onGroupClick(group)}
          >
            <span className="text-[10px] font-semibold text-text-secondary uppercase tracking-wide">
              {group.comboName}
            </span>
            <div className="flex gap-0.5">
              {groupCards.map((card) => (
                <div key={card.id} onClick={(e) => { e.stopPropagation(); onToggleCard(card.id); }}>
                  <CardComponent card={card} size="sm" selected={selectedIds.has(card.id)} />
                </div>
              ))}
            </div>
          </div>
        );
      })}

      {/* Separator if there are groups */}
      {groups.length > 0 && ungroupedCards.length > 0 && (
        <div className="w-px h-16 bg-border mx-1" />
      )}

      {/* Ungrouped cards with rank stacking */}
      {ungroupedGroups.map((rankGroup) => {
        const anySelected = rankGroup.some((c) => selectedIds.has(c.id));
        return (
          <div
            key={rankGroup[0].rank + "-" + rankGroup[0].suit}
            className={`flex flex-col items-center transition-transform ${
              anySelected ? "-translate-y-3" : ""
            }`}
          >
            <CardComponent
              card={rankGroup[0]}
              selected={selectedIds.has(rankGroup[0].id)}
              onClick={() => onToggleCard(rankGroup[0].id)}
            />
            {rankGroup.slice(1).map((card) => (
              <div
                key={card.id}
                data-card-id={card.id}
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
                    color: card.suit === 1 || card.suit === 2 ? "#C75D4A" : "#1A1612",
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
