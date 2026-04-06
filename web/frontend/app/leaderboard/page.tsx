"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePlayer } from "@/hooks/usePlayer";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface HumanEntry {
  username: string;
  elo: number;
  games_played: number;
  is_bot: false;
}

interface BotEntry {
  username: string;
  elo: number;
  games_played: null;
  is_bot: true;
}

interface LeaderboardData {
  humans: HumanEntry[];
  bots: BotEntry[];
}

const NAME_WIDTHS = ["w-1/2", "w-2/3", "w-3/5", "w-2/5", "w-3/4", "w-1/2", "w-3/5", "w-2/3"];

function LeaderboardSkeleton() {
  return (
    <div className="max-w-2xl mx-auto px-4 py-6 lg:px-8 lg:py-8 animate-pulse">
      <div className="h-8 w-36 bg-border rounded-lg mb-6" />
      <div className="bg-surface border border-border rounded-xl overflow-hidden">
        <div className="grid grid-cols-[2rem_1fr_4rem] px-4 py-3 border-b border-border gap-2">
          <div className="h-3 w-3 bg-border/50 rounded" />
          <div className="h-3 w-12 bg-border/50 rounded" />
          <div className="h-3 w-6 bg-border/50 rounded" />
        </div>
        {NAME_WIDTHS.map((w, i) => (
          <div key={i} className="grid grid-cols-[2rem_1fr_4rem] px-4 py-3 border-b border-border last:border-0 gap-2 items-center">
            <div className="h-4 w-4 bg-border/40 rounded" />
            <div className={`h-4 ${w} bg-border/40 rounded`} />
            <div className="h-4 w-8 bg-border/40 rounded" />
          </div>
        ))}
      </div>
    </div>
  );
}

export default function LeaderboardPage() {
  const { player } = usePlayer();
  const [data, setData] = useState<LeaderboardData | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/leaderboard`)
      .then((r) => r.json())
      .then(setData)
      .catch(() => {});
  }, []);

  if (!data) {
    return <LeaderboardSkeleton />;
  }

  const merged = [...data.humans, ...data.bots].sort((a, b) => b.elo - a.elo);
  let humanRank = 0;

  return (
    <div className="max-w-2xl mx-auto px-4 py-6 lg:px-8 lg:py-8">
      <h1 className="text-2xl font-bold text-foreground mb-6">Leaderboard</h1>

      <div className="bg-surface border border-border rounded-xl overflow-hidden">
        {/* Header */}
        <div className="grid grid-cols-[2rem_1fr_4rem] sm:grid-cols-[2rem_1fr_4rem_5rem] px-4 py-3 border-b border-border">
          <span className="text-xs font-semibold text-text-secondary uppercase tracking-wide">#</span>
          <span className="text-xs font-semibold text-text-secondary uppercase tracking-wide">Player</span>
          <span className="text-xs font-semibold text-text-secondary uppercase tracking-wide">Elo</span>
          <span className="text-xs font-semibold text-text-secondary uppercase tracking-wide hidden sm:block">Games</span>
        </div>

        {merged.map((entry) => {
          if (entry.is_bot) {
            return (
              <div
                key={entry.username}
                className="grid grid-cols-[2rem_1fr_4rem] sm:grid-cols-[2rem_1fr_4rem_5rem] px-4 py-3 border-b border-border last:border-0 items-center cursor-default"
              >
                <span className="text-sm text-text-secondary">—</span>
                <span className="text-sm text-text-secondary flex items-center gap-2 truncate">
                  {entry.username}
                  <span className="text-xs border border-border text-foreground/50 bg-background px-1.5 py-0.5 rounded-md shrink-0">bot</span>
                </span>
                <span className="text-sm tabular-nums text-text-secondary">{entry.elo}</span>
                <span className="text-sm text-text-secondary hidden sm:block">—</span>
              </div>
            );
          }
          const rank = ++humanRank;
          const isMe = player?.username === entry.username;
          return (
            <Link
              key={entry.username}
              href={`/profile/${entry.username}`}
              className="grid grid-cols-[2rem_1fr_4rem] sm:grid-cols-[2rem_1fr_4rem_5rem] px-4 py-3 border-b border-border hover:bg-background transition-colors items-center"
            >
              <span className="text-sm text-text-secondary tabular-nums">{rank}</span>
              <span className={`text-sm font-medium flex items-center gap-2 truncate ${isMe ? "text-accent" : "text-foreground"}`}>
                {entry.username}
                {isMe && (
                  <span className="text-xs bg-accent/10 text-accent px-1.5 py-0.5 rounded-md font-semibold shrink-0">you</span>
                )}
              </span>
              <span className="text-sm tabular-nums text-foreground">{entry.elo}</span>
              <span className="text-sm tabular-nums text-text-secondary hidden sm:block">{entry.games_played}</span>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
