"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { CardDTO, CardGroup } from "@/lib/types";
import { groupByRank } from "@/lib/cards";
import CardComponent from "./CardComponent";

interface PlayerHandProps {
  cards: CardDTO[];
  selectedIds: Set<string>;
  onToggleCard: (id: string) => void;
  onToggleCards: (ids: string[]) => void;
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
  onToggleCards,
  groups,
  groupedCardIds,
  onGroupClick,
  hiddenIds,
  compact,
}: PlayerHandProps) {
  // Ungrouped cards (not in any group). Flying cards stay in the array so
  // the hand container keeps its width and the layout doesn't reflow mid-animation.
  const ungroupedCards = useMemo(
    () => cards.filter((c) => !groupedCardIds.has(c.id)),
    [cards, groupedCardIds]
  );
  const ungroupedGroups = useMemo(() => groupByRank(ungroupedCards), [ungroupedCards]);

  const handleRankCardClick = (card: CardDTO, rankGroup: CardDTO[]) => {
    if (rankGroup.length > 2 && !rankGroup.some((c) => selectedIds.has(c.id))) {
      onToggleCards(rankGroup.map((c) => c.id));
    } else {
      onToggleCard(card.id);
    }
  };

  // Pill dimensions for rank stacking
  const pillClass = compact ? "w-8 h-[14px]" : "w-9 h-[22px] lg:w-10 lg:h-[26px]";

  // Lock in the tallest height ever rendered (e.g. when groups with labels are present)
  // so the container never shrinks as cards are played, preventing layout shifts on the gameboard.
  const containerRef = useRef<HTMLDivElement>(null);
  const [lockedMinH, setLockedMinH] = useState(0);
  useEffect(() => { setLockedMinH(0); }, [compact]);
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver(([entry]) => {
      setLockedMinH((prev) => Math.max(prev, entry.borderBoxSize[0].blockSize));
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return (
    // overflow-x-auto creates a scroll container. CSS forces overflow-y to auto too,
    // which would clip the -translate-y-3 upward lift. pt-4 provides clearance above cards + hover.
    <div
      ref={containerRef}
      data-testid="player-hand"
      className="overflow-x-auto pb-1 -mx-2"
      style={{ minHeight: lockedMinH || (compact ? 68 : 96) }}
    >
      <div className="flex items-end justify-center gap-3 min-w-max px-2 pt-4">
        {/* Groups on the left */}
        {groups.map((group) => {
          // Keep flying groups in the DOM (invisible) so the hand width stays stable.
          const groupIsHidden = hiddenIds ? group.cardIds.some((cid) => hiddenIds.has(cid)) : false;
          const groupCards = cards.filter((c) => group.cardIds.includes(c.id));
          const isSelected = group.cardIds.some((cid) => selectedIds.has(cid));
          return (
            <div
              key={group.id}
              className={`flex flex-col items-center gap-1 p-2 rounded-lg cursor-pointer transition-colors ${
                isSelected
                  ? "border-2 border-accent bg-[#FFF8F5]"
                  : "border border-border bg-surface hover:border-accent/50"
              } ${groupIsHidden ? "invisible" : ""}`}
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
          <div className={`w-px ${compact ? "h-8" : "h-16"} bg-border mx-1`} />
        )}

        {/* Ungrouped cards with rank stacking */}
        {ungroupedGroups.map((rankGroup) => {
          const anySelected = rankGroup.some((c) => selectedIds.has(c.id));
          const rankGroupIsHidden = hiddenIds ? rankGroup.some((c) => hiddenIds.has(c.id)) : false;
          return (
            <div
              key={rankGroup[0].id}
              className={`flex flex-col items-center transition-transform ${
                anySelected ? "-translate-y-3" : ""
              } ${rankGroupIsHidden ? "invisible" : ""}`}
            >
              <CardComponent
                card={rankGroup[0]}
                size="sm"
                selected={selectedIds.has(rankGroup[0].id)}
                onClick={() => handleRankCardClick(rankGroup[0], rankGroup)}
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
                  onClick={() => handleRankCardClick(card, rankGroup)}
                >
                  <span
                    className={`${compact ? "text-[9px]" : "text-[13px]"} font-semibold`}
                    style={{
                      color: card.rank >= 16
                        ? card.rank === 17 ? "#C75D4A" : "#1A1612"
                        : card.suit === 1 || card.suit === 2 ? "#C75D4A" : "#1A1612",
                    }}
                  >
                    {card.rank >= 16 ? card.rank_display : card.suit_symbol}
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
