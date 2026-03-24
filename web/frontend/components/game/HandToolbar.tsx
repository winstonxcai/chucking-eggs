"use client";

import { useEffect, useRef, useState } from "react";
import type { CardDTO } from "@/lib/types";

interface HandToolbarProps {
  onFlushSelect: (cards: CardDTO[]) => void;
  sfBySuit: Record<number, { label: string; cards: CardDTO[] }[]>;
  onGroup: () => void;
  onUngroup: () => void;
  canGroup: boolean;
  canUngroup: boolean;
}

const SUITS = [
  { suit: 0, symbol: "\u2660", color: "#1A1612" },
  { suit: 1, symbol: "\u2665", color: "#C75D4A" },
  { suit: 2, symbol: "\u2666", color: "#C75D4A" },
  { suit: 3, symbol: "\u2663", color: "#1A1612" },
];

export default function HandToolbar({
  onFlushSelect,
  sfBySuit,
  onGroup,
  onUngroup,
  canGroup,
  canUngroup,
}: HandToolbarProps) {
  const [openSuit, setOpenSuit] = useState<number | null>(null);
  const popoverRef = useRef<HTMLDivElement>(null);

  // Close popover on outside click
  useEffect(() => {
    if (openSuit === null) return;
    const handler = (e: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target as Node)) {
        setOpenSuit(null);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [openSuit]);

  return (
    <div className="flex items-center justify-center gap-4">
      {/* Group / Ungroup */}
      <div className="flex gap-2">
        <button
          className={`px-3.5 py-1.5 rounded-md text-xs font-medium border transition-colors ${
            canGroup
              ? "bg-surface border-border text-foreground hover:border-accent cursor-pointer"
              : "bg-background border-border text-text-secondary cursor-not-allowed opacity-50"
          }`}
          onClick={onGroup}
          disabled={!canGroup}
        >
          Group
        </button>
        <button
          className={`px-3.5 py-1.5 rounded-md text-xs font-medium border transition-colors ${
            canUngroup
              ? "bg-surface border-border text-foreground hover:border-accent cursor-pointer"
              : "bg-background border-border text-text-secondary cursor-not-allowed opacity-50"
          }`}
          onClick={onUngroup}
          disabled={!canUngroup}
        >
          Ungroup
        </button>
      </div>

      <div className="w-px h-5 bg-border" />

      {/* Straight Flush Finder */}
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-text-secondary">Straight Flush</span>
        {SUITS.map(({ suit, symbol, color }) => {
          const sfs = sfBySuit[suit] ?? [];
          const hasSF = sfs.length > 0;
          const isOpen = openSuit === suit;

          return (
            <div key={suit} className="relative" ref={isOpen ? popoverRef : undefined}>
              <button
                className={`px-2.5 py-1 bg-surface border border-border rounded-md transition-colors ${
                  hasSF
                    ? "cursor-pointer hover:border-accent"
                    : "opacity-50 cursor-not-allowed"
                }`}
                onClick={() => hasSF && setOpenSuit(isOpen ? null : suit)}
                disabled={!hasSF}
              >
                <span className="text-[13px]" style={{ color }}>
                  {symbol}
                </span>
              </button>

              {/* Popover */}
              {isOpen && sfs.length > 0 && (
                <div className="absolute bottom-full mb-1.5 left-1/2 -translate-x-1/2 bg-surface border border-border rounded-lg shadow-md p-1.5 min-w-[100px] z-10">
                  {sfs.map((sf, i) => (
                    <button
                      key={i}
                      className="w-full text-left px-2.5 py-1.5 text-xs font-medium text-foreground rounded hover:bg-background transition-colors cursor-pointer whitespace-nowrap"
                      onClick={() => onFlushSelect(sf.cards)}
                    >
                      {sf.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
