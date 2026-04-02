"use client";

import { X } from "lucide-react";
import type { GameOverMsg } from "@/lib/types";

interface GameOverModalProps {
  data: GameOverMsg;
  humanSeat: number;
  onPlayAgain: () => void;
  isMultiplayer?: boolean;
  onSaveState: () => void;
  onDismiss: () => void;
}

export default function GameOverModal({ data, humanSeat, onPlayAgain, isMultiplayer, onSaveState, onDismiss }: GameOverModalProps) {
  const humanReward = data.rewards[humanSeat] ?? 0;
  const won = humanReward > 0;
  const eloChange = data.elo_changes?.[String(humanSeat)];

  return (
    <div
      className="fixed inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-50"
      onClick={onDismiss}
    >
      <div
        className="relative bg-surface rounded-2xl shadow-xl p-5 lg:p-8 max-w-sm w-full flex flex-col items-center gap-4 max-h-[90dvh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onDismiss}
          className="absolute top-3 right-3 text-text-secondary hover:text-foreground transition-colors"
          aria-label="Close"
        >
          <X size={16} />
        </button>
        <span className="text-2xl font-bold text-foreground">
          {won ? "🎉 Victory!" : "💀 Defeat"}
        </span>
        {eloChange ? (
          <div className="flex items-center gap-2">
            <span className="text-sm text-text-secondary">
              {eloChange.before} → {eloChange.after}
            </span>
            <span className={`text-sm font-semibold ${eloChange.delta >= 0 ? "text-team-green" : "text-team-red"}`}>
              {eloChange.delta >= 0 ? "+" : ""}{eloChange.delta}
            </span>
          </div>
        ) : (
          <span className="text-sm text-text-secondary">
            {humanReward > 0 ? "+" : ""}{humanReward} pts
          </span>
        )}

        <div className="w-full flex flex-col gap-1.5 py-2">
          {data.players.map((p, i) => (
            <div key={p.seat} className="flex items-center gap-3">
              <span className="text-sm font-semibold text-text-secondary w-8">
                {i + 1}
                {i === 0 ? "st" : i === 1 ? "nd" : i === 2 ? "rd" : "th"}
              </span>
              <span
                className={`text-sm font-semibold ${
                  p.seat % 2 === humanSeat % 2 ? "text-accent" : "text-foreground"
                }`}
              >
                {p.name}
              </span>
            </div>
          ))}
        </div>

        <button
          className="px-8 py-2.5 bg-gradient-to-b from-accent to-accent-hover text-white rounded-lg font-semibold text-sm shadow-sm hover:opacity-90 transition-opacity duration-150 cursor-pointer"
          onClick={onPlayAgain}
        >
          {isMultiplayer ? "Rematch" : "Play Again"}
        </button>
        <button
          className="text-xs text-border hover:text-text-secondary transition-colors duration-150"
          onClick={onSaveState}
        >
          Save state JSON
        </button>
      </div>
    </div>
  );
}
