export const DIFFICULTY_INFO: Record<
  string,
  { label: string; description: string; emoji: string }
> = {
  easy: {
    label: "Easy",
    description: "Plays it safe. Great for learning.",
    emoji: "\ud83d\udc28",
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
  expert: {
    label: "Expert",
    description: "Neural network. Our strongest AI.",
    emoji: "\ud83d\udc09",
  },
};
