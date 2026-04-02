// ELOs are calibrated Glicko-2 ratings derived from a 13-bot round-robin
// WR matrix (200 games/matchup, 31,200 total games). See runs/wr_matrix_v2/.

// Two groups shown as separate sections in the UI.
export const OUR_BOTS = ["easy", "medium", "competition", "casual", "hard", "master"] as const;
// liuzha and hulalala hidden — port bugs cause them to lose to random
export const COMPETITION_BOTS = ["wjsd", "yaoji", "jidan"] as const;

export const DIFFICULTY_INFO: Record<
  string,
  { label: string; description: string; emoji: string; elo: number }
> = {
  wjsd: {
    label: "Wjsd",
    description: "SAU 3rd Prize \u00b7 2020 NJUPT entry.",
    emoji: "🦉",
    elo: 1212,
  },
  liuzha: {
    label: "Liuzha",
    description: "SEU 2nd Prize \u00b7 2020 NJUPT entry.",
    emoji: "🦋",
    elo: 1260,
  },
  hulalala: {
    label: "Hulalala",
    description: "SEU 3rd Prize \u00b7 2020 NJUPT entry.",
    emoji: "🐸",
    elo: 1264,
  },
  easy: {
    label: "Easy",
    description: "Plays it safe. Great for learning.",
    emoji: "\ud83d\udc28",
    elo: 1415,
  },
  medium: {
    label: "Medium",
    description: "Plans ahead. A real challenge.",
    emoji: "\ud83e\udd8a",
    elo: 1461,
  },
  competition: {
    label: "Competition",
    description: "SEU 1st Prize \u00b7 Li Jing (2020).",
    emoji: "\ud83c\udfc6",
    elo: 1464,
  },
  casual: {
    label: "Casual",
    description: "Straights-first style. A step up.",
    emoji: "\ud83d\udc3c",
    elo: 1523,
  },
  hard: {
    label: "Hard",
    description: "Reads the table. Tough to beat.",
    emoji: "\ud83e\udd81",
    elo: 1621,
  },
  master: {
    label: "Master",
    description: "Fudan 2nd Prize \u00b7 Chen Yuguan (2020).",
    emoji: "\ud83c\udf1f",
    elo: 1726,
  },
  yaoji: {
    label: "Yaoji",
    description: "NUAA 3rd Prize \u00b7 2020 NJUPT entry.",
    emoji: "✈️",
    elo: 1772,
  },
  jidan: {
    label: "Jidan",
    description: "NUAA 2nd Prize \u00b7 2020 NJUPT entry.",
    emoji: "🚀",
    elo: 1779,
  },
};
