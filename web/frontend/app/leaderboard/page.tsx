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
    return (
      <div className="p-8 flex items-center gap-2">
        <div className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="max-w-2xl px-8 py-8">
      <h1 className="text-2xl font-bold text-foreground mb-6">Leaderboard</h1>

      <div className="bg-surface border border-border rounded-xl overflow-hidden">
        {/* Header */}
        <div className="grid grid-cols-[2rem_1fr_5rem_5rem_5rem] px-4 py-2.5 border-b border-border">
          {["#", "Player", "Elo", "Games", "Win%"].map((h) => (
            <span key={h} className="text-xs font-semibold text-text-secondary uppercase tracking-wide">
              {h}
            </span>
          ))}
        </div>

        {/* Human rows */}
        {data.humans.map((entry, i) => {
          const isMe = player?.username === entry.username;
          return (
            <Link
              key={entry.username}
              href={`/profile/${entry.username}`}
              className="grid grid-cols-[2rem_1fr_5rem_5rem_5rem] px-4 py-3 border-b border-border hover:bg-background transition-colors items-center"
            >
              <span className="text-sm text-text-secondary tabular-nums">{i + 1}</span>
              <span className={`text-sm font-medium flex items-center gap-2 ${isMe ? "text-accent" : "text-foreground"}`}>
                {entry.username}
                {isMe && (
                  <span className="text-xs bg-accent/10 text-accent px-1.5 py-0.5 rounded-full font-semibold">you</span>
                )}
              </span>
              <span className="text-sm tabular-nums text-foreground">{entry.elo}</span>
              <span className="text-sm tabular-nums text-text-secondary">{entry.games_played}</span>
              <span className="text-sm tabular-nums text-text-secondary">—</span>
            </Link>
          );
        })}

        {/* Divider */}
        <div className="px-4 py-2 bg-background border-b border-border">
          <span className="text-xs font-semibold text-text-secondary uppercase tracking-wide">AI Anchors</span>
        </div>

        {/* Bot rows */}
        {data.bots.map((bot) => (
          <div
            key={bot.username}
            className="grid grid-cols-[2rem_1fr_5rem_5rem_5rem] px-4 py-3 border-b border-border last:border-0 items-center cursor-default"
          >
            <span className="text-sm text-text-secondary">—</span>
            <span className="text-sm text-text-secondary flex items-center gap-2">
              {bot.username}
              <span className="text-xs border border-border text-text-secondary px-1.5 py-0.5 rounded-full">bot</span>
            </span>
            <span className="text-sm tabular-nums text-text-secondary">{bot.elo}</span>
            <span className="text-sm text-text-secondary">—</span>
            <span className="text-sm text-text-secondary">—</span>
          </div>
        ))}
      </div>
    </div>
  );
}
