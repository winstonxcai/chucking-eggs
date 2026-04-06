"use client";

import { useEffect, useState } from "react";
import { useAnimate } from "framer-motion";
import { LayoutList, X } from "lucide-react";
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
  // Mobile compact controls
  compact?: boolean;
  canGroup?: boolean;
  canUngroup?: boolean;
  onGroup?: () => void;
  onUngroup?: () => void;
  onCombos?: () => void;
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
  compact,
  canGroup,
  canUngroup,
  onGroup,
  onUngroup,
  onCombos,
}: GameControlsProps) {
  const [urgent, setUrgent] = useState(false);
  const [scope, animate] = useAnimate();

  useEffect(() => {
    if (!isMyTurn || !turnDeadlineMs) {
      setUrgent(false);
      return;
    }
    const remaining = Math.max(0, turnDeadlineMs - Date.now());
    if (remaining <= 15000) {
      setUrgent(true);
      return;
    }
    setUrgent(false);
    const t = setTimeout(() => setUrgent(true), remaining - 15000);
    return () => clearTimeout(t);
  }, [isMyTurn, turnDeadlineMs]);

  useEffect(() => {
    if (!scope.current) return;
    if (!isMyTurn || !turnDeadlineMs) {
      animate(scope.current, { width: "100%" }, { duration: 0 });
      return;
    }
    const remaining = Math.max(0, turnDeadlineMs - Date.now());
    const initialPct = Math.min(100, Math.max(0, (remaining / (TIMEOUT_S * 1000)) * 100));
    const controls = animate(
      scope.current,
      [{ width: `${initialPct}%` }, { width: "0%" }],
      { duration: remaining / 1000, ease: "linear" }
    );
    return () => controls.cancel();
  }, [isMyTurn, turnDeadlineMs]);

  const showRope = isMyTurn && !!turnDeadlineMs;

  if (compact) {
    return (
      <div className="flex flex-col gap-1">
        {/* Thin rope timer — 1/4 width, centered */}
        <div className={`flex justify-center ${!showRope ? "invisible" : ""}`}>
          <div
            data-testid="rope-timer"
            className="w-1/4 h-0.5 rounded-full overflow-hidden bg-border"
          >
            <div
              ref={scope}
              className={`h-full rounded-full ${urgent ? "bg-team-red" : "bg-accent"}`}
            />
          </div>
        </div>
        {/* Single row: Group/Ungroup | Play | × | Pass | Combos */}
        <div className="flex items-center justify-center gap-1.5">
          <button
            className={`px-2 py-1 rounded-md text-xs font-medium border transition-colors ${
              canGroup || canUngroup
                ? "bg-surface border-border text-foreground hover:border-accent cursor-pointer"
                : "bg-background border-border text-text-secondary cursor-not-allowed opacity-50"
            }`}
            onClick={canUngroup ? onUngroup : onGroup}
            disabled={!canGroup && !canUngroup}
          >
            {canUngroup ? "Ungroup" : "Group"}
          </button>

          <div className={`flex items-center gap-1.5 ${!isMyTurn ? "invisible" : ""}`}>
            <button
              className={`px-3 py-1 rounded-lg text-xs font-semibold transition-colors ${
                matchingCombo
                  ? "bg-accent text-white hover:bg-accent-hover cursor-pointer"
                  : "bg-border text-text-secondary cursor-not-allowed"
              }`}
              disabled={!matchingCombo}
              onClick={onPlay}
            >
              {matchingCombo ? "Play" : "Select"}
            </button>
            <button
              className={`px-2 py-1 rounded-lg text-xs font-medium border-[1.5px] border-border text-text-secondary hover:border-foreground hover:text-foreground transition-colors cursor-pointer${hasSelection ? "" : " invisible"}`}
              onClick={onUnselect}
            >
              <X size={14} strokeWidth={2} />
            </button>
            <button
              className={`px-2.5 py-1 rounded-lg text-xs font-medium border-[1.5px] border-border text-text-secondary hover:border-foreground hover:text-foreground transition-colors cursor-pointer ${isLeading ? "invisible" : ""}`}
              onClick={onPass}
            >
              Pass
            </button>
          </div>

          <button
            className="flex items-center gap-1 px-2 py-1 bg-surface border border-border rounded-md text-xs font-medium text-foreground cursor-pointer"
            onClick={onCombos}
          >
            <LayoutList size={12} />
            Combos
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center gap-2">
      {/* Rope timer — always rendered to reserve space, invisible when not your turn */}
      <div
        data-testid="rope-timer"
        className={`w-48 h-1 bg-border rounded-full overflow-hidden ${!showRope ? "invisible" : ""}`}
      >
        <div
          ref={scope}
          className={`h-full rounded-full ${urgent ? "bg-team-red" : "bg-accent"}`}
        />
      </div>

      <div className={`flex items-center justify-center gap-3 ${!isMyTurn ? "invisible" : ""}`}>
        <button
          className={`px-7 py-2.5 rounded-lg text-sm font-semibold transition-colors min-w-[160px] ${
            matchingCombo
              ? "bg-accent text-white hover:bg-accent-hover cursor-pointer"
              : "bg-border text-text-secondary cursor-not-allowed"
          }`}
          disabled={!matchingCombo}
          onClick={onPlay}
        >
          {matchingCombo ? `Play ${matchingCombo.type_name}` : "Select"}
        </button>
        <button
          className={`px-4 py-2.5 rounded-lg text-sm font-medium border-[1.5px] border-border transition-colors ${
            hasSelection
              ? "text-text-secondary hover:border-foreground hover:text-foreground cursor-pointer"
              : "text-text-secondary opacity-50 cursor-not-allowed"
          }`}
          disabled={!hasSelection}
          onClick={onUnselect}
        >
          Unselect
        </button>
        <button
          className={`px-7 py-2.5 rounded-lg text-sm font-medium border-[1.5px] border-border transition-colors ${
            isLeading
              ? "text-text-secondary opacity-50 cursor-not-allowed"
              : "text-text-secondary hover:border-foreground hover:text-foreground cursor-pointer"
          }`}
          disabled={isLeading}
          onClick={onPass}
        >
          Pass
        </button>
      </div>
    </div>
  );
}
