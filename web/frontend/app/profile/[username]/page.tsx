"use client";

import { Suspense, useEffect, useState } from "react";
import { useParams } from "next/navigation";
import EloChart from "@/components/profile/EloChart";
import GameFeed from "@/components/profile/GameFeed";
import StatsGrid from "@/components/profile/StatsGrid";
import StatusScreen from "@/components/layout/StatusScreen";

function ProfileSkeleton() {
  return (
    <div className="max-w-2xl mx-auto px-8 py-8 flex flex-col gap-6 animate-pulse">
      <div className="bg-surface border border-border rounded-xl px-6 py-5 flex items-center justify-between">
        <div className="flex flex-col gap-2">
          <div className="h-7 w-32 bg-border rounded-lg" />
          <div className="h-4 w-48 bg-border/60 rounded" />
        </div>
        <div className="flex flex-col items-end gap-1">
          <div className="h-7 w-12 bg-border rounded-lg" />
          <div className="h-3 w-6 bg-border/60 rounded" />
        </div>
      </div>
      <div className="bg-surface border border-border rounded-xl px-6 py-5">
        <div className="h-4 w-24 bg-border rounded mb-4" />
        <div className="h-36 bg-border/40 rounded-lg" />
      </div>
      <div className="bg-surface border border-border rounded-xl px-6 py-5">
        <div className="h-4 w-44 bg-border rounded mb-4" />
        <div className="grid grid-cols-3 gap-3">
          <div className="h-16 bg-border/40 rounded-lg" />
          <div className="h-16 bg-border/40 rounded-lg" />
          <div className="h-16 bg-border/40 rounded-lg" />
        </div>
      </div>
      <div className="bg-surface border border-border rounded-xl px-6 py-5">
        <div className="h-4 w-28 bg-border rounded mb-4" />
        {[0, 1, 2, 3, 4].map((i) => (
          <div key={i} className="h-10 bg-border/40 rounded-lg mb-2 last:mb-0" />
        ))}
      </div>
    </div>
  );
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface PlayerDoc {
  _id: string;
  username: string;
  elo: number;
  games_played: number;
}

interface ProfileData {
  player: PlayerDoc;
  games: unknown[];
  elo_history: { date: string; elo: number }[];
}

function ProfileContent() {
  const params = useParams();
  const username = typeof params.username === "string" ? params.username : "";

  const [data, setData] = useState<ProfileData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!username) return;
    fetch(`${API_BASE}/api/profile/${username}`)
      .then((r) => {
        if (!r.ok) throw new Error("Player not found");
        return r.json();
      })
      .then(setData)
      .catch((e) => setError(e.message));
  }, [username]);

  if (error) {
    return (
      <StatusScreen
        variant="info"
        title="Player not found"
        message={`We couldn't find a player named "${username}". They may not have an account yet.`}
        action={{ label: "Leaderboard", href: "/leaderboard" }}
        secondaryAction={{ label: "Home", href: "/" }}
      />
    );
  }

  if (!data) {
    return <ProfileSkeleton />;
  }

  const { player, games, elo_history = [] } = data;

  // Compute win rate
  const wins = (games as { players: { display_name: string; is_bot: boolean; team_result: string }[] }[])
    .filter((g) => g.players.find((p) => !p.is_bot && p.display_name === username)?.team_result === "win")
    .length;
  const total = player.games_played;
  const winPct = total > 0 ? Math.round((wins / total) * 100) : 0;

  return (
    <div className="max-w-2xl mx-auto px-8 py-8 flex flex-col gap-6">
      {/* Header */}
      <div className="bg-surface border border-border rounded-xl px-6 py-5 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">{player.username}</h1>
          <p className="text-sm text-text-secondary mt-0.5">
            {total} game{total === 1 ? "" : "s"} · {winPct}% win rate
          </p>
        </div>
        <div className="flex flex-col items-end">
          <span className="text-2xl font-bold text-accent">{player.elo}</span>
          <span className="text-xs text-text-secondary">Elo</span>
        </div>
      </div>

      {/* Elo History */}
      <div className="bg-surface border border-border rounded-xl px-6 py-5">
        <h2 className="text-sm font-semibold text-foreground mb-4">Elo History</h2>
        <EloChart data={elo_history} />
      </div>

      {/* Win Rate by Difficulty */}
      <div className="bg-surface border border-border rounded-xl px-6 py-5">
        <h2 className="text-sm font-semibold text-foreground mb-4">Win Rate by Difficulty</h2>
        <StatsGrid games={games as Parameters<typeof StatsGrid>[0]["games"]} username={username} />
      </div>

      {/* Recent Games */}
      <div className="bg-surface border border-border rounded-xl px-6 py-5">
        <h2 className="text-sm font-semibold text-foreground mb-2">Recent Games</h2>
        <GameFeed games={games as Parameters<typeof GameFeed>[0]["games"]} username={username} />
      </div>
    </div>
  );
}

export default function ProfilePage() {
  return (
    <Suspense fallback={<ProfileSkeleton />}>
      <ProfileContent />
    </Suspense>
  );
}
