"use client";

const DIFFICULTY_ORDER = [
  "ez", "wjsd", "random", "hulalala", "liuzha",
  "greedy", "lalala", "heuristic", "xingdream", "strategic", "noai",
  "yaoji", "jidan",
  // legacy keys (old game records)
  "easy", "casual", "hard", "competition", "master",
];

const DIFFICULTY_LABEL: Record<string, string> = {
  ez: "Ez", wjsd: "Wjsd", random: "Random",
  hulalala: "Hulalala", liuzha: "Liuzha",
  greedy: "Greedy", lalala: "Lalala", heuristic: "Heuristic",
  xingdream: "Xingdream", strategic: "Strategic", noai: "NoAI",
  yaoji: "Yaoji", jidan: "Jidan",
  // legacy
  easy: "Easy (legacy)", casual: "Casual (legacy)",
  hard: "Hard (legacy)", competition: "Competition (legacy)", master: "Master (legacy)",
};

interface GameDoc {
  mode: string;
  difficulty: string | null;
  players: { display_name: string; is_bot: boolean; team_result: string }[];
}

interface StatsGridProps {
  games: GameDoc[];
  username: string;
}

export default function StatsGrid({ games, username }: StatsGridProps) {
  // Build per-difficulty stats
  const stats: Record<string, { wins: number; total: number }> = {};
  let favDiff = "";
  let favCount = 0;

  for (const game of games) {
    if (!game.difficulty || game.mode !== "solo") continue;
    const diff = game.difficulty;
    if (!stats[diff]) stats[diff] = { wins: 0, total: 0 };
    const me = game.players.find((p) => !p.is_bot && p.display_name === username);
    if (!me) continue;
    stats[diff].total++;
    if (me.team_result === "win") stats[diff].wins++;
    if (stats[diff].total > favCount) {
      favCount = stats[diff].total;
      favDiff = diff;
    }
  }

  const rows = DIFFICULTY_ORDER.filter((d) => stats[d]);
  if (rows.length === 0) return <p className="text-sm text-text-secondary">No solo games yet.</p>;

  return (
    <div className="flex flex-col gap-3">
      {rows.map((diff) => {
        const { wins, total } = stats[diff];
        const pct = total > 0 ? wins / total : 0;
        return (
          <div key={diff} className="flex items-center gap-3">
            <span className="text-xs text-text-secondary w-20 shrink-0">
              {DIFFICULTY_LABEL[diff] ?? diff}
            </span>
            <div className="flex-1 h-1.5 bg-border rounded-full overflow-hidden">
              <div
                className="h-full bg-accent rounded-full transition-[width]"
                style={{ width: `${pct * 100}%` }}
              />
            </div>
            <span className="text-xs text-text-secondary tabular-nums w-16 text-right shrink-0">
              {wins}W {total - wins}L
            </span>
          </div>
        );
      })}
      {favDiff && (
        <p className="text-xs text-text-secondary mt-1">
          Most played: {DIFFICULTY_LABEL[favDiff] ?? favDiff} ({favCount} games)
        </p>
      )}
    </div>
  );
}
