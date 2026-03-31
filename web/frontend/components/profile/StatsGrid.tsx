"use client";

const DIFFICULTY_ORDER = [
  "easy", "medium", "casual", "hard", "expert", "master",
  "yaoji", "jidan", "competition", "wjsd", "liuzha", "hulalala",
];

const DIFFICULTY_LABEL: Record<string, string> = {
  easy: "Easy",
  medium: "Medium",
  casual: "Casual",
  hard: "Hard",
  expert: "Expert",
  master: "Master",
  yaoji: "Yaoji",
  jidan: "Jidan",
  competition: "Competition",
  wjsd: "Wjsd",
  liuzha: "Liuzha",
  hulalala: "Hulalala",
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
              {wins}W / {total}G
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
