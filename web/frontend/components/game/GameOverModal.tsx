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

const ORDINALS = ["1st", "2nd", "3rd", "4th"];
const ORDINAL_COLORS = [
  "text-amber-500",
  "text-zinc-400",
  "text-amber-700/70",
  "text-text-secondary",
];

export default function GameOverModal({ data, humanSeat, onPlayAgain, isMultiplayer, onSaveState, onDismiss }: GameOverModalProps) {
  const humanReward = data.rewards[humanSeat] ?? 0;
  const won = humanReward > 0;
  const myElo = data.elo_changes?.[String(humanSeat)];

  // Team finish positions (e.g. "1-3" means teammates finished 1st and 3rd)
  const teamPositions = data.finish_order
    .map((seat, i) => ({ seat, pos: i + 1 }))
    .filter((p) => p.seat % 2 === humanSeat % 2)
    .map((p) => p.pos)
    .sort((a, b) => a - b);
  const teamLabel = teamPositions.join("-");

  return (
    <div
      className="fixed inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-50"
      onClick={onDismiss}
    >
      <div
        className="relative bg-surface rounded-2xl shadow-xl p-6 lg:p-8 max-w-sm w-full mx-4 flex flex-col gap-5 max-h-[90dvh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onDismiss}
          className="absolute top-3 right-3 text-text-secondary hover:text-foreground transition-colors"
          aria-label="Close"
        >
          <X size={16} />
        </button>

        {/* Result header */}
        <div className="text-center pt-1">
          <div className={`text-3xl font-bold tracking-tight ${won ? "text-accent" : "text-foreground"}`}>
            {won ? "Victory" : "Defeat"}
          </div>
          <div className="text-sm text-text-secondary mt-1">
            {teamLabel}
          </div>
        </div>

        <div className="w-full h-px bg-border" />

        {/* Leaderboard — elo shown inline per player for duo/quad */}
        <div className="w-full flex flex-col divide-y divide-border">
          {data.players.map((p, i) => {
            const elo = data.elo_changes?.[String(p.seat)];
            const isMe = p.seat === humanSeat;
            const isTeammate = p.seat % 2 === humanSeat % 2 && !isMe;
            return (
              <div key={p.seat} className="flex items-center gap-3 py-2.5">
                <span className={`text-xs font-semibold w-7 shrink-0 tabular-nums ${ORDINAL_COLORS[i]}`}>
                  {ORDINALS[i]}
                </span>
                <span className={`flex-1 text-sm font-semibold truncate ${
                  isMe ? "text-accent" : isTeammate ? "text-team-green" : "text-foreground"
                }`}>
                  {p.name}
                  {isMe && (
                    <span className="text-xs font-normal text-text-secondary ml-1.5">(you)</span>
                  )}
                  {isTeammate && (
                    <span className="text-xs font-normal text-text-secondary ml-1.5">(partner)</span>
                  )}
                </span>
                {elo ? (
                  <span className={`text-xs font-semibold tabular-nums shrink-0 ${
                    elo.delta >= 0 ? "text-team-green" : "text-team-red"
                  }`}>
                    {elo.delta >= 0 ? "+" : ""}{elo.delta}
                  </span>
                ) : (
                  <span className="w-8 shrink-0" />
                )}
              </div>
            );
          })}
        </div>

        {/* Viewer's full elo summary (before → after) */}
        {myElo && (
          <>
            <div className="w-full h-px bg-border" />
            <div className="flex items-center justify-between text-sm">
              <span className="text-text-secondary">Rating</span>
              <span className="text-text-secondary tabular-nums">
                {myElo.before} → {myElo.after}
              </span>
            </div>
          </>
        )}

        {/* Actions */}
        <button
          className="w-full py-2.5 bg-gradient-to-b from-accent to-accent-hover text-white rounded-lg font-semibold text-sm shadow-sm hover:opacity-90 transition-opacity duration-150 cursor-pointer"
          onClick={onPlayAgain}
        >
          {isMultiplayer ? "Rematch" : "Play Again"}
        </button>
        <button
          className="text-xs text-border hover:text-text-secondary transition-colors duration-150 -mt-2"
          onClick={onSaveState}
        >
          Save state JSON
        </button>
      </div>
    </div>
  );
}
