import Link from "next/link";

const OUR_BOTS = [
  { name: "Easy",   elo: 1415, source: "Strategic heuristic agent" },
  { name: "Medium", elo: 1523, source: "Strategic heuristic agent" },
  { name: "Hard",   elo: 1621, source: "Strategic heuristic agent" },
];

const COMPETITION_BOTS = [
  { name: "Wjsd",        elo: 1212, source: "SAU",   award: "3rd Prize" },
  { name: "Competition", elo: 1464, source: "SEU",   award: "1st Prize · Li Jing" },
  { name: "Master",      elo: 1726, source: "Fudan", award: "2nd Prize · Chen Yuguan" },
  { name: "Yaoji",       elo: 1772, source: "NUAA",  award: "3rd Prize" },
  { name: "Jidan",       elo: 1779, source: "NUAA",  award: "2nd Prize" },
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
        <div className="flex items-center gap-4">
          <Link
            href="/rules"
            className="inline-flex items-center gap-1.5 text-sm font-medium text-accent hover:text-accent-hover transition-colors"
          >
            Full rules
            <span aria-hidden="true">→</span>
          </Link>
          <Link
            href="https://www.pagat.com/climbing/guan_dan.html"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-sm text-text-secondary hover:text-foreground transition-colors"
          >
            Pagat.com reference
            <span aria-hidden="true">↗</span>
          </Link>
        </div>
      </section>

      {/* Bot Opponents */}
      <section className="flex flex-col gap-6">
        <div className="flex flex-col gap-1">
          <h2 className="text-xl font-bold text-foreground tracking-tight">Bot Opponents</h2>
          <p className="text-xs text-text-secondary">
            Elo ratings calibrated via a 31,200-game round-robin (200 games per matchup).
          </p>
        </div>

        {/* Our Bots */}
        <div className="flex flex-col gap-2">
          <h3 className="text-sm font-semibold text-foreground">Our Bots</h3>
          <div className="border border-border rounded-xl overflow-hidden">
            <div className="grid grid-cols-[112px_56px_1fr] gap-3 px-4 py-2.5 bg-background border-b border-border">
              <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider">Bot</span>
              <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider text-right">Elo</span>
              <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider">Type</span>
            </div>
            {OUR_BOTS.map((bot, i) => (
              <div
                key={bot.name}
                className={`grid grid-cols-[112px_56px_1fr] gap-3 px-4 py-3 items-center ${
                  i < OUR_BOTS.length - 1 ? "border-b border-border" : ""
                }`}
              >
                <span className="text-sm font-semibold text-foreground">{bot.name}</span>
                <span className="text-sm font-mono text-text-secondary text-right">{bot.elo}</span>
                <span className="text-sm text-text-secondary">{bot.source}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Competition Bots */}
        <div className="flex flex-col gap-2">
          <div className="flex flex-col gap-0.5">
            <h3 className="text-sm font-semibold text-foreground">Competition Bots</h3>
            <p className="text-xs text-text-secondary">
              Ported from the 2020 NJUPT Guan Dan AI Competition with permission.
              NJUPT = Nanjing University of Posts and Telecommunications.
            </p>
          </div>
          <div className="border border-border rounded-xl overflow-hidden">
            <div className="grid grid-cols-[112px_56px_1fr] gap-3 px-4 py-2.5 bg-background border-b border-border">
              <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider">Bot</span>
              <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider text-right">Elo</span>
              <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider">University · Award</span>
            </div>
            {COMPETITION_BOTS.map((bot, i) => (
              <div
                key={bot.name}
                className={`grid grid-cols-[112px_56px_1fr] gap-3 px-4 py-3 items-center ${
                  i < COMPETITION_BOTS.length - 1 ? "border-b border-border" : ""
                }`}
              >
                <span className="text-sm font-semibold text-foreground">{bot.name}</span>
                <span className="text-sm font-mono text-text-secondary text-right">{bot.elo}</span>
                <span className="text-sm text-text-secondary">{bot.source} · {bot.award}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* This Project */}
      <section className="flex flex-col gap-2">
        <h2 className="text-xl font-bold text-foreground tracking-tight">This Project</h2>
        <p className="text-sm text-text-secondary leading-relaxed">
          Chucking Eggs is an open-source Guan Dan engine and deep reinforcement learning trainer.
          The strategic and heuristic bots were built in-house.
          Competition bots were ported from the 2020 NJUPT Guan Dan AI Competition with permission.
        </p>
      </section>
    </div>
  );
}
