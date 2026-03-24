"use client";

import type { GameOverMsg } from "@/lib/types";

interface GameOverModalProps {
  data: GameOverMsg;
  humanSeat: number;
  onPlayAgain: () => void;
}

export default function GameOverModal({ data, humanSeat, onPlayAgain }: GameOverModalProps) {
  const humanReward = data.rewards[humanSeat] ?? 0;
  const won = humanReward > 0;

  return (
    <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-50">
      <div className="bg-surface rounded-2xl shadow-xl p-8 max-w-sm w-full flex flex-col items-center gap-4">
        <span className="text-2xl font-bold text-foreground">
          {won ? "Victory!" : "Defeat"}
        </span>
        <span className="text-sm text-text-secondary">
          Level change: {humanReward > 0 ? "+" : ""}
          {humanReward}
        </span>

        <div className="w-full flex flex-col gap-1.5 py-2">
          {data.players.map((p, i) => (
            <div key={p.seat} className="flex items-center gap-3">
              <span className="text-xs font-medium text-text-secondary w-8">
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
          className="px-8 py-2.5 bg-accent text-white rounded-lg font-semibold text-sm hover:bg-accent-hover transition-colors cursor-pointer"
          onClick={onPlayAgain}
        >
          Play Again
        </button>
      </div>
    </div>
  );
}
