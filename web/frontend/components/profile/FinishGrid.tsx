"use client";

interface GamePlayer {
  display_name: string;
  is_bot: boolean;
  seat: number;
  finish_pos: number;
}

interface GameDoc {
  players: GamePlayer[];
}

interface FinishGridProps {
  games: GameDoc[];
  username: string;
}

const WIN_PAIRS = [
  { key: "1,2", label: "1st · 2nd", color: "#1F5C42" },
  { key: "1,3", label: "1st · 3rd", color: "#2D7A5A" },
  { key: "1,4", label: "1st · 4th", color: "#3D8C6F" },
  { key: "2,1", label: "2nd · 1st", color: "#58A080" },
  { key: "3,1", label: "3rd · 1st", color: "#7BB89A" },
  { key: "4,1", label: "4th · 1st", color: "#A2CFBE" },
];

const LOSS_PAIRS = [
  { key: "2,3", label: "2nd · 3rd", color: "#E5A99F" },
  { key: "2,4", label: "2nd · 4th", color: "#D98F82" },
  { key: "3,2", label: "3rd · 2nd", color: "#C97062" },
  { key: "3,4", label: "3rd · 4th", color: "#B85848" },
  { key: "4,2", label: "4th · 2nd", color: "#A4412F" },
  { key: "4,3", label: "4th · 3rd", color: "#8B2D1D" },
];

export default function FinishGrid({ games, username }: FinishGridProps) {
  const counts: Record<string, number> = {};

  for (const game of games) {
    const me = game.players.find((p) => !p.is_bot && p.display_name === username);
    if (!me || me.finish_pos === -1) continue;
    const partnerSeat = (me.seat + 2) % 4;
    const partner = game.players.find((p) => p.seat === partnerSeat);
    if (!partner || partner.finish_pos === -1) continue;
    const key = `${me.finish_pos},${partner.finish_pos}`;
    counts[key] = (counts[key] ?? 0) + 1;
  }

  const winTotal = WIN_PAIRS.reduce((s, p) => s + (counts[p.key] ?? 0), 0);
  const lossTotal = LOSS_PAIRS.reduce((s, p) => s + (counts[p.key] ?? 0), 0);

  if (winTotal === 0 && lossTotal === 0) {
    return <p className="text-sm text-text-secondary">No finish data yet.</p>;
  }

  return (
    <div className="flex flex-col gap-5">
      {winTotal > 0 && (
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-xs font-medium text-text-secondary uppercase tracking-wide">Wins</span>
            <span className="text-xs tabular-nums text-text-secondary">{winTotal}</span>
          </div>
          <div className="flex w-full h-3 rounded-full overflow-hidden">
            {WIN_PAIRS.map(({ key, color }) =>
              (counts[key] ?? 0) > 0 ? (
                <div
                  key={key}
                  style={{ width: `${((counts[key] ?? 0) / winTotal) * 100}%`, backgroundColor: color }}
                />
              ) : null
            )}
          </div>
          <div className="grid grid-cols-3 gap-x-4 gap-y-1.5 mt-2.5">
            {WIN_PAIRS.filter((p) => (counts[p.key] ?? 0) > 0).map(({ key, label, color }) => (
              <span key={key} className="flex items-center gap-1.5 text-xs text-text-secondary">
                <span className="w-2 h-2 rounded-sm shrink-0" style={{ backgroundColor: color }} />
                {label} {counts[key]}
              </span>
            ))}
          </div>
        </div>
      )}

      {lossTotal > 0 && (
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-xs font-medium text-text-secondary uppercase tracking-wide">Losses</span>
            <span className="text-xs tabular-nums text-text-secondary">{lossTotal}</span>
          </div>
          <div className="flex w-full h-3 rounded-full overflow-hidden">
            {LOSS_PAIRS.map(({ key, color }) =>
              (counts[key] ?? 0) > 0 ? (
                <div
                  key={key}
                  style={{ width: `${((counts[key] ?? 0) / lossTotal) * 100}%`, backgroundColor: color }}
                />
              ) : null
            )}
          </div>
          <div className="grid grid-cols-3 gap-x-4 gap-y-1.5 mt-2.5">
            {LOSS_PAIRS.filter((p) => (counts[p.key] ?? 0) > 0).map(({ key, label, color }) => (
              <span key={key} className="flex items-center gap-1.5 text-xs text-text-secondary">
                <span className="w-2 h-2 rounded-sm shrink-0" style={{ backgroundColor: color }} />
                {label} {counts[key]}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
