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
  compact,
}: PlayerHandProps) {
  // Ungrouped cards (not in any group, not flying)
  const ungroupedCards = useMemo(
    () => cards.filter((c) => !groupedCardIds.has(c.id) && !hiddenIds?.has(c.id)),
    [cards, groupedCardIds, hiddenIds]
  );
  const ungroupedGroups = useMemo(() => groupByRank(ungroupedCards), [ungroupedCards]);

  // Pill dimensions for rank stacking
  const pillClass = compact ? "w-8 h-[14px]" : "w-9 h-[22px]";

  return (
    // overflow-x-auto creates a scroll container. CSS forces overflow-y to auto too,
    // which would clip the -translate-y-3 upward lift. pt-3 provides clearance above cards.
    <div className="overflow-x-auto pb-1 -mx-2">
      <div className="flex items-end justify-center gap-3 min-w-max px-2 pt-3">
        {/* Groups on the left */}
        {groups.map((group) => {
          // Hide group if any of its cards are flying
          if (hiddenIds && group.cardIds.some((cid) => hiddenIds.has(cid))) return null;
          const groupCards = cards.filter((c) => group.cardIds.includes(c.id));
          const isSelected = group.cardIds.some((cid) => selectedIds.has(cid));
          return (
            <div
              key={group.id}
              className={`flex flex-col items-center gap-1 p-2 rounded-lg cursor-pointer transition-colors ${
                isSelected
                  ? "border-2 border-accent bg-[#FFF8F5]"
                  : "border border-border bg-surface hover:border-accent/50"
              }`}
              onClick={() => onGroupClick(group)}
            >
              <span className="text-[10px] font-semibold text-text-secondary uppercase tracking-wide">
                {group.comboName}
              </span>
              <div className="flex gap-0.5">
                {groupCards.map((card) => (
                  <CardComponent key={card.id} card={card} size={compact ? "xs" : "sm"} />
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
                size={compact ? "sm" : "md"}
                selected={selectedIds.has(rankGroup[0].id)}
                onClick={() => onToggleCard(rankGroup[0].id)}
              />
              {rankGroup.slice(1).map((card) => (
                <div
                  key={card.id}
                  data-card-id={card.id}
                  className={`${pillClass} flex items-center justify-center rounded cursor-pointer -mt-1 ${
                    selectedIds.has(card.id)
                      ? "bg-surface border-2 border-accent shadow-[0_1px_4px_rgba(217,119,87,0.15)]"
                      : "bg-surface border-[1.5px] border-border shadow-[0_1px_2px_rgba(0,0,0,0.06)]"
                  }`}
                  onClick={() => onToggleCard(card.id)}
                >
                  <span
                    className={`${compact ? "text-[9px]" : "text-[13px]"} font-semibold`}
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
    </div>
  );
}
