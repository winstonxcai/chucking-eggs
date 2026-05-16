export const OUR_BOTS = ["random", "greedy", "heuristic", "strategic"] as const;

export type BotInfo = { label: string; description: string; elo: number };
