"use client";

import type { CardDTO, PlayerDTO } from "@/lib/types";
import { groupByRank } from "@/lib/cards";
import CardComponent from "./CardComponent";

interface OpponentPanelProps {
  player: PlayerDTO;
  thinking?: boolean;
  revealedHand?: CardDTO[];
  isActive?: boolean;
  isReviewMode?: boolean;
}

export default function OpponentPanel({ player, thinking, revealedHand, isActive, isReviewMode }: OpponentPanelProps) {
  const teamColor = player.is_teammate ? "border-l-team-green" : "border-l-team-red";
  const pulseClass = player.is_teammate ? "animate-border-pulse-green" : "animate-border-pulse-red";
  const borderColor = `${teamColor}${isActive ? ` ${pulseClass}` : ""}`;
  const roleLabel = player.is_out ? "Out" : player.is_teammate ? "Partner" : "Opp";
  const statusLabel = player.is_out
    ? "Out"
    : isReviewMode
      ? `${player.card_count} card${player.card_count === 1 ? "" : "s"}`
      : player.is_teammate
        ? "Partner"
        : "Opp";

  return (
    <div
      className={`relative flex flex-col gap-0.5 px-1.5 py-1 lg:px-3 lg:py-2 min-w-[60px] lg:min-w-[96px] bg-surface rounded-xl border border-border border-l-[3px] ${borderColor} shadow-[0_1px_3px_rgba(0,0,0,0.06)]`}
    >
      {/* Mobile: single-line "Name · Role" */}
      <div className="lg:hidden flex items-center gap-1">
        <span className="text-[10px] font-semibold text-foreground truncate max-w-[52px] relative">
          {thinking ? (
            <>
              <span className="invisible">{player.name}</span>
              <span className="absolute inset-0 flex items-center animate-pulse">···</span>
            </>
          ) : player.name}
        </span>
        <span className="text-[10px] text-text-secondary">· {roleLabel}</span>
      </div>

      {/* Desktop: two-line name + status */}
      <div className="hidden lg:flex flex-col gap-0.5">
        <div className="flex items-center gap-1.5">
          <span className="text-sm font-semibold text-foreground relative whitespace-nowrap">
            {thinking ? (
              <>
                <span className="invisible">{player.name}</span>
                <span className="absolute inset-0 flex items-center animate-pulse text-text-secondary">···</span>
              </>
            ) : player.name}
          </span>
        </div>
        <span className="text-xs text-text-secondary">{statusLabel}</span>
      </div>

      {revealedHand && revealedHand.length > 0 && (
        <div
          data-testid={`opponent-hand-${player.seat}`}
          className="hidden lg:block mt-1"
        >
          <div className="flex gap-x-0.5 gap-y-0">
            {groupByRank(revealedHand).map((rankGroup) => (
              <div key={rankGroup[0].id} className="flex flex-col items-center">
                <CardComponent card={rankGroup[0]} size="sm" />
                {rankGroup.slice(1).map((card) => (
                  <div
                    key={card.id}
                    className="w-9 h-[22px] lg:w-10 lg:h-[26px] flex items-center justify-center rounded -mt-1 bg-surface border-[1.5px] border-border shadow-[0_1px_2px_rgba(0,0,0,0.06)]"
                  >
                    <span
                      className="text-[13px] font-semibold"
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
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
