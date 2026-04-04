const COMBOS = [
  { name: "Single",     desc: "One card. Higher rank wins." },
  { name: "Pair",       desc: "Two cards of the same rank." },
  { name: "Triple",     desc: "Three cards of the same rank." },
  { name: "Full House", desc: "A triple plus any pair. Ranked by the triple." },
  { name: "Straight",   desc: "Exactly five consecutive cards. Aces high (10-J-Q-K-A) or low (A-2-3-4-5)." },
  { name: "Tube",       desc: "Exactly three consecutive pairs — e.g. 7-7-8-8-9-9." },
  { name: "Plate",      desc: "Exactly two consecutive triples — e.g. 7-7-7-8-8-8." },
];

const BOMBS = [
  { name: "Quad",              desc: "Four of a kind.", note: "lowest" },
  { name: "Five–Ten of a kind", desc: "Larger same-rank sets beat smaller ones.", note: "" },
  { name: "Straight Flush",    desc: "Five consecutive cards, all the same suit.", note: "" },
  { name: "Four Jokers",       desc: "Both red jokers + both black jokers.", note: "beats everything" },
];

function TableRow({ name, desc, note, last }: { name: string; desc: string; note?: string; last: boolean }) {
  return (
    <div className={`grid grid-cols-[160px_1fr] gap-3 px-4 py-3 items-start ${last ? "" : "border-b border-border"}`}>
      <div className="flex flex-col gap-0.5">
        <span className="text-sm font-semibold text-foreground">{name}</span>
        {note && <span className="text-xs text-text-secondary italic">{note}</span>}
      </div>
      <span className="text-sm text-text-secondary">{desc}</span>
    </div>
  );
}

export default function RulesPage() {
  return (
    <div className="max-w-2xl mx-auto px-6 py-10 flex flex-col gap-10">

      {/* Intro */}
      <section className="flex flex-col gap-3">
        <h1 className="text-2xl font-bold text-foreground tracking-tight">How to Play Guan Dan</h1>
        <p className="text-sm text-text-secondary leading-relaxed">
          The name 掼蛋 (guàn dàn) translates roughly to &ldquo;throwing eggs&rdquo; — a name as
          irreverent as the game is addictive. Guan Dan is a partnership card game from Jiangsu
          province in China. Four players compete in two teams of two, working together to empty
          their hands before the other side. At its core it&apos;s a climbing game: play a
          combination, and your opponents have to top it. But there are bombs. And wildcards. And
          your teammate who somehow has all the aces.
        </p>
      </section>

      {/* Setup */}
      <section className="flex flex-col gap-3">
        <h2 className="text-xl font-bold text-foreground tracking-tight">The Setup</h2>
        <ul className="text-sm text-text-secondary leading-relaxed flex flex-col gap-1.5 list-disc list-inside">
          <li>4 players, 2 teams — partners sit across the table from each other</li>
          <li>Two standard decks + 4 jokers = 108 cards, 27 per player</li>
        </ul>
      </section>

      {/* Card Rankings */}
      <section className="flex flex-col gap-3">
        <h2 className="text-xl font-bold text-foreground tracking-tight">Card Rankings</h2>
        <p className="text-sm text-text-secondary leading-relaxed">
          From highest to lowest:
        </p>
        <p className="text-sm font-mono text-foreground bg-background border border-border rounded-lg px-4 py-3 leading-relaxed">
          Red Joker &gt; Black Joker &gt; <strong>2</strong> &gt; A &gt; K &gt; Q &gt; J &gt; 10 &gt; 9 &gt; 8 &gt; 7 &gt; 6 &gt; 5 &gt; 4 &gt; 3
        </p>
        <p className="text-sm text-text-secondary leading-relaxed">
          2s rank above aces because 2 is the <em>level card</em> for this game — more on that below.
        </p>
      </section>

      {/* How a Round Works */}
      <section className="flex flex-col gap-3">
        <h2 className="text-xl font-bold text-foreground tracking-tight">How a Round Works</h2>
        <p className="text-sm text-text-secondary leading-relaxed">
          One player leads by playing any valid combination face-up. Play continues counterclockwise.
          On your turn you must either beat the current combination with a higher one of the same
          type, play any bomb, or pass. Three consecutive passes end the trick — the last player to
          play leads the next one. This continues until players run out of cards.
        </p>
      </section>

      {/* What You Can Play */}
      <section className="flex flex-col gap-4">
        <div className="flex flex-col gap-1">
          <h2 className="text-xl font-bold text-foreground tracking-tight">What You Can Play</h2>
          <p className="text-sm text-text-secondary">
            Seven ordinary combinations. You can only beat a combination with a higher one of the
            same type — unless you&apos;re throwing a bomb.
          </p>
        </div>
        <div className="border border-border rounded-xl overflow-hidden">
          <div className="grid grid-cols-[160px_1fr] gap-3 px-4 py-2.5 bg-background border-b border-border">
            <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider">Hand</span>
            <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider">What it is</span>
          </div>
          {COMBOS.map((combo, i) => (
            <div
              key={combo.name}
              className={`grid grid-cols-[160px_1fr] gap-3 px-4 py-3 items-start ${i < COMBOS.length - 1 ? "border-b border-border" : ""}`}
            >
              <span className="text-sm font-semibold text-foreground">{combo.name}</span>
              <span className="text-sm text-text-secondary">{combo.desc}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Bombs */}
      <section className="flex flex-col gap-4">
        <div className="flex flex-col gap-1">
          <h2 className="text-xl font-bold text-foreground tracking-tight">Bombs</h2>
          <p className="text-sm text-text-secondary">
            Any bomb beats any ordinary combination. Higher bomb types beat lower ones. Within the
            same type, higher rank wins.
          </p>
        </div>
        <div className="border border-border rounded-xl overflow-hidden">
          <div className="grid grid-cols-[160px_1fr] gap-3 px-4 py-2.5 bg-background border-b border-border">
            <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider">Bomb</span>
            <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider">What it is</span>
          </div>
          {BOMBS.map((bomb, i) => (
            <TableRow
              key={bomb.name}
              name={bomb.name}
              desc={bomb.desc}
              note={bomb.note || undefined}
              last={i === BOMBS.length - 1}
            />
          ))}
        </div>
      </section>

      {/* Wildcards */}
      <section className="flex flex-col gap-3">
        <h2 className="text-xl font-bold text-foreground tracking-tight">Wildcards: The 2♥</h2>
        <p className="text-sm text-text-secondary leading-relaxed">
          The 2 of Hearts (and its twin from the second deck) are wild — they can stand in for any
          non-joker card. Use them to fill gaps in a straight, round out a full house, complete a
          tube. One restriction: wild 2♥s can&apos;t substitute inside bombs.
        </p>
      </section>

      {/* Winning */}
      <section className="flex flex-col gap-3">
        <h2 className="text-xl font-bold text-foreground tracking-tight">Winning</h2>
        <p className="text-sm text-text-secondary leading-relaxed">
          The first team to have both players&apos; hands empty wins the round. Margins matter — a
          1st-and-2nd finish earns more than 1st-and-3rd. In the full game, teams climb through
          levels 2 → A, advancing faster with bigger wins. Level progression is coming soon.
        </p>
      </section>

    </div>
  );
}
