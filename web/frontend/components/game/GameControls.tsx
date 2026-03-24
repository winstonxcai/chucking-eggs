"use client";

import type { ComboDTO } from "@/lib/types";

interface GameControlsProps {
  matchingCombo: ComboDTO | null;
  isLeading: boolean;
  isMyTurn: boolean;
  onPlay: () => void;
  onPass: () => void;
}

export default function GameControls({
  matchingCombo,
  isLeading,
  isMyTurn,
  onPlay,
  onPass,
}: GameControlsProps) {
  if (!isMyTurn) return null;

  return (
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
      {!isLeading && (
        <button
          className="px-7 py-2.5 rounded-lg text-sm font-medium border-[1.5px] border-border text-text-secondary hover:border-foreground hover:text-foreground transition-colors cursor-pointer"
          onClick={onPass}
        >
          Pass
        </button>
      )}
    </div>
  );
}
