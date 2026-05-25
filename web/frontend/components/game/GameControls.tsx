"use client";

import { useEffect, useRef, useState } from "react";
import { animate } from "framer-motion";
import { LayoutList, X } from "lucide-react";
import type { ComboDTO } from "@/lib/types";

const TIMEOUT_S = 30;

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
  const [countdown, setCountdown] = useState<number | null>(null);
  const timerRef = useRef<HTMLDivElement>(null);

  const stableDeadline = isMyTurn ? turnDeadlineMs ?? null : null;

  // Urgency + countdown via setInterval — runs even in background tabs (rAF pauses there)
  useEffect(() => {
    if (!isMyTurn || !stableDeadline) {
      setUrgent(false);
      setCountdown(null);
      return;
    }
    const tick = () => {
      const remaining = Math.max(0, stableDeadline - Date.now());
      if (remaining <= 15000) {
        setUrgent(true);
        setCountdown(Math.ceil(remaining / 1000));
      } else {
        setUrgent(false);
        setCountdown(null);
      }
    };
    tick();
    const id = setInterval(tick, 500);
    return () => clearInterval(id);
  }, [isMyTurn, stableDeadline]);

  // Rope bar animation — scaleX is GPU-composited (no layout reflow), runs at 60 fps
  useEffect(() => {
    if (!timerRef.current) return;
    if (!isMyTurn || !stableDeadline) {
      animate(timerRef.current, { scaleX: 1 }, { duration: 0 });
      return;
    }
    const remaining = Math.max(0, stableDeadline - Date.now());
    const initialFraction = Math.min(1, Math.max(0, remaining / (TIMEOUT_S * 1000)));
    const controls = animate(
      timerRef.current,
      { scaleX: [initialFraction, 0] },
      { duration: remaining / 1000, ease: "linear" }
    );
    return () => controls.cancel();
  }, [isMyTurn, stableDeadline]);

  const showRope = isMyTurn && !!stableDeadline;
  const countdownLabel =
    urgent && countdown !== null
      ? isLeading
        ? `Auto-play in ${countdown}s`
        : `Auto-pass in ${countdown}s`
      : null;

  if (compact) {
    return (
      <div className="flex flex-col gap-1">
        {/* Thin rope timer — 1/4 width, centered */}
        <div className={`flex flex-col items-center ${!showRope ? "invisible" : ""}`}>
          <div
            data-testid="rope-timer"
            className="w-1/4 h-0.5 rounded-full overflow-hidden bg-border"
          >
            <div
              ref={timerRef}
              className={`h-full rounded-full origin-left ${urgent ? "bg-team-red" : "bg-accent"}`}
            />
          </div>
          {countdownLabel && (
            <span className="text-[10px] text-team-red font-medium mt-0.5">
              {countdownLabel}
            </span>
          )}
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
              data-testid="play-button"
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
              data-testid="unselect-button"
              className={`px-2 py-1 rounded-lg text-xs font-medium border-[1.5px] border-border text-text-secondary hover:border-foreground hover:text-foreground transition-colors cursor-pointer${hasSelection ? "" : " invisible"}`}
              onClick={onUnselect}
            >
              <X size={14} strokeWidth={2} />
            </button>
            <button
              data-testid="pass-button"
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
      <div className={`flex flex-col items-center ${!showRope ? "invisible" : ""}`}>
        <div
          data-testid="rope-timer"
          className="w-48 h-1 bg-border rounded-full overflow-hidden"
        >
          <div
            ref={timerRef}
            className={`h-full rounded-full origin-left ${urgent ? "bg-team-red" : "bg-accent"}`}
          />
        </div>
        {countdownLabel && (
          <span className="text-xs text-team-red font-medium mt-1">
            {countdownLabel}
          </span>
        )}
      </div>

      <div className={`flex items-center justify-center gap-3 ${!isMyTurn ? "invisible" : ""}`}>
        <button
          data-testid="play-button"
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
          data-testid="unselect-button"
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
          data-testid="pass-button"
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
