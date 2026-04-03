"use client";

import type { CardDTO, PlayerDTO } from "@/lib/types";
import CardComponent from "./CardComponent";

interface OpponentPanelProps {
  player: PlayerDTO;
  thinking?: boolean;
  revealedHand?: CardDTO[];
}

export default function OpponentPanel({ player, thinking, revealedHand }: OpponentPanelProps) {
  const borderColor = player.is_teammate ? "border-l-team-green" : "border-l-team-red";

  return (
    <div
      className={`flex flex-col items-center gap-1 px-1.5 py-1.5 lg:px-4 lg:py-3 bg-surface rounded-xl border-l-[3px] ${borderColor} shadow-[0_1px_3px_rgba(0,0,0,0.06)]`}
    >
      {/* Avatar: desktop only */}
      <div className="hidden lg:flex w-9 h-9 rounded-full bg-[#F0EDE8] items-center justify-center">
        {thinking ? (
          <span className="animate-pulse text-xs text-text-secondary">...</span>
        ) : (
          <span className="text-sm text-text-secondary">
            {player.is_teammate ? "\u2660" : "\u2666"}
          </span>
        )}
      </div>
      <div className="flex flex-col items-center gap-0.5">
        <span className="text-[11px] lg:text-sm font-semibold text-foreground truncate max-w-[56px] lg:max-w-none">
          {thinking ? <span className="animate-pulse">...</span> : player.name}
        </span>
        {/* Mobile: compact role only */}
        <span className="lg:hidden text-[10px] text-text-secondary">
          {player.is_out ? "Out" : player.is_teammate ? "Partner" : "Opp"}
        </span>
        {/* Desktop: card count + role */}
        <span className="hidden lg:block text-xs text-text-secondary">
          {player.is_out
            ? "Out"
            : player.card_count <= 10
              ? `${player.card_count} cards`
              : player.is_teammate
                ? "Partner"
                : "Opponent"}
        </span>
      </div>
      {revealedHand && revealedHand.length > 0 && (
        <div className="flex flex-wrap gap-0.5 max-w-[200px] justify-center">
          {revealedHand.map((card) => (
            <CardComponent key={card.id} card={card} size="sm" />
          ))}
        </div>
      )}
    </div>
  );
}
