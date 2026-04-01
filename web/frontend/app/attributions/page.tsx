import Link from "next/link";

const BOTS = [
  { name: "Wjsd",        elo: 1212, source: "SAU · 3rd Prize",          event: "2020 NJUPT Guan Dan Competition" },
  { name: "Liuzha",      elo: 1260, source: "SEU · 2nd Prize",          event: "2020 NJUPT Guan Dan Competition" },
  { name: "Hulalala",    elo: 1264, source: "SEU · 3rd Prize",          event: "2020 NJUPT Guan Dan Competition" },
  { name: "Easy",        elo: 1415, source: "Strategic heuristic agent", event: "This project" },
  { name: "Medium",      elo: 1461, source: "Strategic heuristic agent", event: "This project" },
  { name: "Competition", elo: 1464, source: "SEU · 1st Prize · Li Jing", event: "2020 NJUPT Guan Dan Competition" },
  { name: "Casual",      elo: 1523, source: "Strategic heuristic agent", event: "This project" },
  { name: "Hard",        elo: 1621, source: "Strategic heuristic agent", event: "This project" },
  { name: "Master",      elo: 1726, source: "Fudan · 2nd Prize · Chen Yuguan", event: "2020 NJUPT Guan Dan Competition" },
  { name: "Yaoji",       elo: 1772, source: "NUAA · 3rd Prize",         event: "2020 NJUPT Guan Dan Competition" },
  { name: "Jidan",       elo: 1779, source: "NUAA · 2nd Prize",         event: "2020 NJUPT Guan Dan Competition" },
  { name: "Expert",      elo: 1786, source: "Deep RL neural network",   event: "This project" },
];

export default function AttributionsPage() {
  return (
    <div className="max-w-2xl mx-auto px-6 py-10 flex flex-col gap-10">
      {/* How to Play */}
      <section className="flex flex-col gap-3">
        <h1 className="text-2xl font-bold text-foreground tracking-tight">How to Play</h1>
        <p className="text-sm text-text-secondary leading-relaxed">
          Guan Dan is a Chinese trick-taking card game played in teams of two.
          Partners sit across from each other and work together to be the first team
          to have both players finish all their cards.
        </p>
        <Link
          href="https://www.pagat.com/climbing/guan_dan.html"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 text-sm font-medium text-accent hover:text-accent-hover transition-colors"
        >
          Full rules on Pagat.com
          <span aria-hidden="true">→</span>
        </Link>
      </section>

      {/* Bot Opponents */}
      <section className="flex flex-col gap-4">
        <div className="flex flex-col gap-1">
          <h2 className="text-xl font-bold text-foreground tracking-tight">Bot Opponents</h2>
          <p className="text-xs text-text-secondary">
            Elo ratings calibrated via a 31,200-game round-robin (200 games per matchup).
            NJUPT = Nanjing University of Posts and Telecommunications.
          </p>
        </div>

        <div className="border border-border rounded-xl overflow-hidden">
          {/* Header */}
          <div className="grid grid-cols-[1fr_80px_1fr] gap-3 px-4 py-2.5 bg-background border-b border-border">
            <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider">Bot</span>
            <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider text-right">Elo</span>
            <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider">Source</span>
          </div>

          {BOTS.map((bot, i) => (
            <div
              key={bot.name}
              className={`grid grid-cols-[1fr_80px_1fr] gap-3 px-4 py-3 items-start ${
                i < BOTS.length - 1 ? "border-b border-border" : ""
              }`}
            >
              <span className="text-sm font-semibold text-foreground">{bot.name}</span>
              <span className="text-sm font-mono text-text-secondary text-right">{bot.elo}</span>
              <div className="flex flex-col gap-0.5">
                <span className="text-sm text-foreground">{bot.source}</span>
                {bot.event !== "This project" && (
                  <span className="text-xs text-text-secondary">{bot.event}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* This Project */}
      <section className="flex flex-col gap-2">
        <h2 className="text-xl font-bold text-foreground tracking-tight">This Project</h2>
        <p className="text-sm text-text-secondary leading-relaxed">
          Chucking Eggs is an open-source Guan Dan engine and deep reinforcement learning trainer.
          The strategic, heuristic, and Expert bots were built and trained in-house.
          Competition bots were ported from the 2020 NJUPT Guan Dan AI Competition with permission.
        </p>
      </section>
    </div>
  );
}
