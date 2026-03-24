"use client";

interface HandToolbarProps {
  onFlushFind: (suit: number) => void;
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
  onFlushFind,
  onGroup,
  onUngroup,
  canGroup,
  canUngroup,
}: HandToolbarProps) {
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
        {SUITS.map(({ suit, symbol, color }) => (
          <button
            key={suit}
            className="px-2.5 py-1 bg-surface border border-border rounded-md cursor-pointer hover:border-accent transition-colors"
            onClick={() => onFlushFind(suit)}
          >
            <span className="text-[13px]" style={{ color }}>
              {symbol}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
