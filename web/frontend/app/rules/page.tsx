import CardComponent from "@/components/game/CardComponent";
import type { CardDTO } from "@/lib/types";

function c(rank_display: string, suit: number, rank = 1, is_wild = false): CardDTO {
  return {
    rank,
    suit: suit < 4 ? suit : 0,
    deck: 0,
    id: `rule-${rank_display}-${suit}-${rank}`,
    display: rank_display,
    rank_display,
    suit_symbol: (["♠", "♥", "♦", "♣"] as const)[suit] ?? "",
    is_wild,
  };
}

const RANK_ORDER: CardDTO[] = [
  c("RJ", 0, 17), c("BJ", 0, 16),
  c("2", 1, 2), c("A", 0), c("K", 0), c("Q", 0), c("J", 0),
  c("10", 0), c("9", 0), c("8", 0), c("7", 0), c("6", 0), c("5", 0), c("4", 0), c("3", 0),
];

const COMBOS = [
  {
    name: "Single",
    desc: "One card. Higher rank wins.",
    cards: [c("A", 0)],
  },
  {
    name: "Pair",
    desc: "Two cards of the same rank.",
    cards: [c("7", 1), c("7", 2)],
  },
  {
    name: "Triple",
    desc: "Three cards of the same rank.",
    cards: [c("K", 0), c("K", 1), c("K", 3)],
  },
  {
    name: "Full House",
    desc: "A triple plus any pair. Ranked by the triple.",
    cards: [c("9", 0), c("9", 1), c("9", 2), c("4", 1), c("4", 2)],
  },
  {
    name: "Straight",
    desc: "Exactly five consecutive cards. Aces high (10-J-Q-K-A) or low (A-2-3-4-5).",
    cards: [c("7", 0), c("8", 1), c("9", 2), c("10", 3), c("J", 0)],
  },
  {
    name: "Tube",
    desc: "Three consecutive pairs.",
    cards: [c("7", 1), c("7", 2), c("8", 0), c("8", 3), c("9", 1), c("9", 2)],
  },
  {
    name: "Plate",
    desc: "Two consecutive triples.",
    cards: [c("8", 0), c("8", 1), c("8", 2), c("9", 0), c("9", 1), c("9", 3)],
  },
];

const BOMBS = [
  {
    name: "Quad",
    note: "lowest",
    desc: "Four of a kind.",
    cards: [c("3", 0), c("3", 1), c("3", 2), c("3", 3)],
  },
  {
    name: "Straight Flush",
    note: "",
    desc: "Five consecutive same-suit cards.",
    cards: [c("4", 0), c("5", 0), c("6", 0), c("7", 0), c("8", 0)],
  },
  {
    name: "Four Jokers",
    note: "highest",
    desc: "Both black jokers and both red jokers. Unbeatable.",
    cards: [c("BJ", 0, 16), c("BJ", 0, 16), c("RJ", 0, 17), c("RJ", 0, 17)],
  },
];

function Cards({ cards, size = "sm" }: { cards: CardDTO[]; size?: "xs" | "sm" | "md" }) {
  return (
    <div className="flex gap-1.5 flex-wrap">
      {cards.map((card, i) => (
        <CardComponent key={i} card={card} size={size} />
      ))}
    </div>
  );
}

export default function RulesPage() {
  return (
    <div className="max-w-2xl mx-auto px-6 py-10 flex flex-col gap-12">

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
        <ul className="text-sm text-text-secondary leading-relaxed flex flex-col gap-1 list-disc list-inside">
          <li>4 players, 2 teams — partners sit across the table</li>
          <li>Two standard decks + 4 jokers = 108 cards, 27 per player</li>
        </ul>
      </section>

      {/* Card Rankings */}
      <section className="flex flex-col gap-4">
        <h2 className="text-xl font-bold text-foreground tracking-tight">Card Rankings</h2>
        <p className="text-sm text-text-secondary">Highest to lowest:</p>
        <div className="flex gap-1 flex-wrap items-center">
          {RANK_ORDER.map((card, i) => (
            <div key={i} className="flex items-center gap-1">
              <CardComponent card={card} size="xs" />
              {i < RANK_ORDER.length - 1 && (
                <span className="text-[10px] text-text-secondary font-medium">›</span>
              )}
            </div>
          ))}
        </div>
        <p className="text-sm text-text-secondary leading-relaxed">
          2s rank above aces because 2 is the <em>level card</em> for this game. The{" "}
          <span className="font-medium text-foreground">2♥</span> is also a wildcard — more on that below.
        </p>
      </section>

      {/* How a Round Works */}
      <section className="flex flex-col gap-3">
        <h2 className="text-xl font-bold text-foreground tracking-tight">How a Round Works</h2>
        <p className="text-sm text-text-secondary leading-relaxed">
          One player leads by playing any valid combination face-up. Play continues
          counterclockwise. On your turn you must either beat the current combination with a higher
          one of the same type, play any bomb, or pass. Three consecutive passes end the trick —
          the last player to play leads the next one. This continues until players run out of cards.
        </p>
      </section>

      {/* What You Can Play */}
      <section className="flex flex-col gap-6">
        <div className="flex flex-col gap-1">
          <h2 className="text-xl font-bold text-foreground tracking-tight">What You Can Play</h2>
          <p className="text-sm text-text-secondary">
            Seven ordinary combinations. You can only beat a combination with a higher one of the
            same type — unless you&apos;re throwing a bomb.
          </p>
        </div>
        <div className="flex flex-col gap-6">
          {COMBOS.map((combo) => (
            <div key={combo.name} className="flex flex-col gap-2">
              <span className="text-sm font-semibold text-foreground">{combo.name}</span>
              <Cards cards={combo.cards} size="sm" />
              <p className="text-xs text-text-secondary">{combo.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Bombs */}
      <section className="flex flex-col gap-6">
        <div className="flex flex-col gap-1">
          <h2 className="text-xl font-bold text-foreground tracking-tight">Bombs</h2>
          <p className="text-sm text-text-secondary">
            Any bomb beats any ordinary combination. Higher bomb types beat lower ones. Within the
            same type, higher rank wins.
          </p>
        </div>
        <div className="flex flex-col gap-6">
          {BOMBS.map((bomb) => (
            <div key={bomb.name} className="flex flex-col gap-2">
              <div className="flex items-baseline gap-2">
                <span className="text-sm font-semibold text-foreground">{bomb.name}</span>
                {bomb.note && (
                  <span className="text-xs text-text-secondary italic">{bomb.note}</span>
                )}
              </div>
              <Cards cards={bomb.cards} size="sm" />
              <p className="text-xs text-text-secondary">{bomb.desc}</p>
            </div>
          ))}
          <p className="text-xs text-text-secondary leading-relaxed">
            Five through ten of a kind also count as bombs, ranked by size — more cards beats fewer.
          </p>
        </div>
      </section>

      {/* Wildcards */}
      <section className="flex flex-col gap-3">
        <h2 className="text-xl font-bold text-foreground tracking-tight">The Wildcard: 2♥</h2>
        <Cards cards={[c("2", 1, 2, true), c("2", 1, 2, true)]} size="sm" />
        <p className="text-sm text-text-secondary leading-relaxed">
          The 2 of Hearts — and its twin from the second deck — are wild. They can stand in for
          any non-joker card: fill the gap in a straight, round out a full house, complete a tube.
          One restriction: wild 2♥s can&apos;t substitute inside bombs.
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
