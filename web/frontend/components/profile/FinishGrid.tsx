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
  { key: "1,2", label: "1st · 2nd", color: "#09402A" },
  { key: "1,3", label: "1st · 3rd", color: "#186340" },
  { key: "1,4", label: "1st · 4th", color: "#2E8A58" },
  { key: "2,1", label: "2nd · 1st", color: "#52AB7A" },
  { key: "3,1", label: "3rd · 1st", color: "#82C79E" },
  { key: "4,1", label: "4th · 1st", color: "#BBE0CC" },
];

const LOSS_PAIRS = [
  { key: "2,3", label: "2nd · 3rd", color: "#F5C8C0" },
  { key: "2,4", label: "2nd · 4th", color: "#E89080" },
  { key: "3,2", label: "3rd · 2nd", color: "#CF5840" },
  { key: "3,4", label: "3rd · 4th", color: "#A83020" },
  { key: "4,2", label: "4th · 2nd", color: "#7C1810" },
  { key: "4,3", label: "4th · 3rd", color: "#4E0A06" },
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
                {label} {counts[key]} ({Math.round((counts[key] ?? 0) / winTotal * 100)}%)
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
                {label} {counts[key]} ({Math.round((counts[key] ?? 0) / lossTotal * 100)}%)
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
