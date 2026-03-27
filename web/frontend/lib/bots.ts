export const DIFFICULTY_INFO: Record<
  string,
  { label: string; description: string; emoji: string; elo: number }
> = {
  easy: {
    label: "Easy",
    description: "Plays it safe. Great for learning.",
    emoji: "\ud83d\udc28",
    elo: 1421,
  },
  wjsd: {
    label: "Wjsd",
    description: "SAU 3rd Prize \u00b7 2020 NJUPT entry.",
    emoji: "\ud83c\udfc5",
    elo: 1250,
  },
  casual: {
    label: "Casual",
    description: "Straights-first style. A step up.",
    emoji: "\ud83d\udc3c",
    elo: 1566,
  },
  medium: {
    label: "Medium",
    description: "Plans ahead. A real challenge.",
    emoji: "\ud83e\udd8a",
    elo: 1492,
  },
  competition: {
    label: "Competition",
    description: "SEU 1st Prize \u00b7 Li Jing (2020).",
    emoji: "\ud83c\udfc6",
    elo: 1550,
  },
  hard: {
    label: "Hard",
    description: "Reads the table. Tough to beat.",
    emoji: "\ud83e\udd81",
    elo: 1703,
  },
  yaoji: {
    label: "Yaoji",
    description: "NUAA 3rd Prize \u00b7 2020 NJUPT entry.",
    emoji: "\ud83c\udfc5",
    elo: 1760,
  },
  jidan: {
    label: "Jidan",
    description: "NUAA 2nd Prize \u00b7 2020 NJUPT entry.",
    emoji: "\ud83c\udfc5",
    elo: 1775,
  },
  expert: {
    label: "Expert",
    description: "Neural network. Our strongest AI.",
    emoji: "\ud83d\udc09",
    elo: 1790,
  },
  hulalala: {
    label: "Hulalala",
    description: "SEU 3rd Prize \u00b7 2020 NJUPT entry.",
    emoji: "\ud83c\udfc5",
    elo: 1820,
  },
  liuzha: {
    label: "Liuzha",
    description: "SEU 2nd Prize \u00b7 2020 NJUPT entry.",
    emoji: "\ud83c\udfc5",
    elo: 1840,
  },
  master: {
    label: "Master",
    description: "Fudan 2nd Prize \u00b7 Chen Yuguan (2020).",
    emoji: "\ud83c\udf1f",
    elo: 1766,
  },
};
