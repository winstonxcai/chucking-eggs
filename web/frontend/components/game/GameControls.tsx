"use client";

import { useEffect, useState } from "react";
import type { ComboDTO } from "@/lib/types";

const TIMEOUT_S = 90;

interface GameControlsProps {
  matchingCombo: ComboDTO | null;
  isLeading: boolean;
  isMyTurn: boolean;
  hasSelection: boolean;
  turnDeadlineMs?: number;
  onPlay: () => void;
  onPass: () => void;
  onUnselect: () => void;
}

export default function GameControls({
  matchingCombo,
  isLeading,
  isMyTurn,
  hasSelection,
  turnDeadlineMs,
  onPlay,
  onPass,
  onUnselect,
}: GameControlsProps) {
  const [remainingMs, setRemainingMs] = useState<number | null>(null);

  useEffect(() => {
    if (!isMyTurn || !turnDeadlineMs) {
      setRemainingMs(null);
      return;
    }
    const tick = () => setRemainingMs(Math.max(0, turnDeadlineMs - Date.now()));
    tick();
    const id = setInterval(tick, 200);
    return () => clearInterval(id);
  }, [isMyTurn, turnDeadlineMs]);

  const showRope = isMyTurn && remainingMs !== null;
  const progress = remainingMs !== null ? remainingMs / (TIMEOUT_S * 1000) : 1;
  const urgent = remainingMs !== null && remainingMs < 15000;

  return (
    <div className={`flex flex-col items-center gap-2 ${!isMyTurn ? "invisible" : ""}`}>
      {/* Rope timer */}
      {showRope && (
        <div data-testid="rope-timer" className="w-48 h-1 bg-border rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-[width] duration-200 ${urgent ? "bg-team-red" : "bg-accent"}`}
            style={{ width: `${progress * 100}%` }}
          />
        </div>
      )}

      <div className="flex items-center justify-center gap-3">
        <button
          className={`px-7 py-2.5 rounded-lg text-sm font-semibold transition-colors ${
            matchingCombo
              ? "bg-accent text-white hover:bg-accent-hover cursor-pointer"
              : "bg-border text-text-secondary cursor-not-allowed"
          }`}
          disabled={!matchingCombo}
          onClick={onPlay}
        >
          {matchingCombo ? `Play ${matchingCombo.type_name}` : "Select cards"}
        </button>
        {hasSelection && (
          <button
            className="px-4 py-2.5 rounded-lg text-sm font-medium border-[1.5px] border-border text-text-secondary hover:border-foreground hover:text-foreground transition-colors cursor-pointer"
            onClick={onUnselect}
          >
            Unselect
          </button>
        )}
        {!isLeading && (
          <button
            className="px-7 py-2.5 rounded-lg text-sm font-medium border-[1.5px] border-border text-text-secondary hover:border-foreground hover:text-foreground transition-colors cursor-pointer"
            onClick={onPass}
          >
            Pass
          </button>
        )}
      </div>
    </div>
  );
}
