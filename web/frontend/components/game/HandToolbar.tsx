"use client";

interface HandToolbarProps {
  onFlushFind: (suit: number) => void;
}

const SUITS = [
  { suit: 0, symbol: "\u2660", color: "#1A1612" },
  { suit: 1, symbol: "\u2665", color: "#C75D4A" },
  { suit: 2, symbol: "\u2666", color: "#C75D4A" },
  { suit: 3, symbol: "\u2663", color: "#1A1612" },
];

export default function HandToolbar({ onFlushFind }: HandToolbarProps) {
  return (
    <div className="flex items-center justify-center gap-4">
      <div className="w-px h-5 bg-border" />
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
