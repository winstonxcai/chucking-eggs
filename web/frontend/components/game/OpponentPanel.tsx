"use client";

import type { CardDTO, PlayerDTO } from "@/lib/types";
import CardComponent from "./CardComponent";

interface OpponentPanelProps {
  player: PlayerDTO;
  thinking?: boolean;
  revealedHand?: CardDTO[];
  isActive?: boolean;
}

export default function OpponentPanel({ player, thinking, revealedHand, isActive }: OpponentPanelProps) {
  const teamColor = player.is_teammate ? "border-l-team-green" : "border-l-team-red";
  const borderColor = `${teamColor}${isActive ? " animate-border-pulse" : ""}`;
  const roleLabel = player.is_out ? "Out" : player.is_teammate ? "Partner" : "Opp";
  const statusLabel = player.is_out
    ? "Out"
    : player.card_count <= 10
      ? `${player.card_count} card${player.card_count === 1 ? "" : "s"}`
      : player.is_teammate
        ? "Partner"
        : "Opponent";

  return (
    <div
      className={`flex flex-col gap-0.5 px-1.5 py-1 lg:px-3 lg:py-2 bg-surface rounded-xl border border-border border-l-[3px] ${borderColor} shadow-[0_1px_3px_rgba(0,0,0,0.06)]`}
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
          <span className="text-sm font-semibold text-foreground relative">
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
        <div className="flex flex-wrap gap-0.5 max-w-[200px] justify-center mt-1">
          {revealedHand.map((card) => (
            <CardComponent key={card.id} card={card} size="sm" />
          ))}
        </div>
      )}
    </div>
  );
}
