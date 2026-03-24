"use client";

import type { PlayerDTO } from "@/lib/types";

interface OpponentPanelProps {
  player: PlayerDTO;
  thinking?: boolean;
}

export default function OpponentPanel({ player, thinking }: OpponentPanelProps) {
  const borderColor = player.is_teammate ? "border-l-team-green" : "border-l-team-red";

  return (
    <div
      className={`flex flex-col items-center gap-2 px-4 py-3 bg-surface rounded-xl border-l-[3px] ${borderColor} shadow-[0_1px_3px_rgba(0,0,0,0.06)]`}
    >
      <div className="w-9 h-9 rounded-full bg-[#F0EDE8] flex items-center justify-center text-lg">
        {thinking ? (
          <span className="animate-pulse text-sm text-text-secondary">...</span>
        ) : (
          <span className="text-sm text-text-secondary">
            {player.is_teammate ? "\u2660" : "\u2666"}
          </span>
        )}
      </div>
      <div className="flex flex-col items-center gap-0.5">
        <div className="flex items-center gap-1.5">
          <span className="text-sm font-semibold text-foreground">{player.name}</span>
          <span className="text-xs font-medium text-text-secondary">{player.elo}</span>
        </div>
        <span className="text-xs text-text-secondary">
          {player.is_out
            ? "Out"
            : player.card_count <= 10
              ? `${player.card_count} cards`
              : player.is_teammate
                ? "Partner"
                : "Opponent"}
        </span>
      </div>
    </div>
  );
}
