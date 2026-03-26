export const DIFFICULTY_INFO: Record<
  string,
  { label: string; description: string; emoji: string }
> = {
  easy: {
    label: "Easy",
    description: "Plays it safe. Great for learning.",
    emoji: "\ud83d\udc28",
  },
  casual: {
    label: "Casual",
    description: "Straights-first style. A step up.",
    emoji: "\ud83d\udc3c",
  },
  medium: {
    label: "Medium",
    description: "Plans ahead. A real challenge.",
    emoji: "\ud83e\udd8a",
  },
  hard: {
    label: "Hard",
    description: "Reads the table. Tough to beat.",
    emoji: "\ud83e\udd81",
  },
  competition: {
    label: "Competition",
    description: "SEU 1st Prize \u00b7 Li Jing (2020).",
    emoji: "\ud83c\udfc6",
  },
  expert: {
    label: "Expert",
    description: "Neural network. Our strongest AI.",
    emoji: "\ud83d\udc09",
  },
  master: {
    label: "Master",
    description: "Fudan 2nd Prize \u00b7 Chen Yuguan (2020).",
    emoji: "\ud83c\udf1f",
  },
};
