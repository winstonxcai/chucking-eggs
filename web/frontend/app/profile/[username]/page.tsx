"use client";

import { Suspense, useEffect, useState } from "react";
import { useParams } from "next/navigation";
import EloChart from "@/components/profile/EloChart";
import GameFeed from "@/components/profile/GameFeed";
import StatsGrid from "@/components/profile/StatsGrid";

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
}

function buildEloHistory(games: unknown[], username: string) {
  const points: { date: string; elo: number }[] = [];
  const sorted = [...(games as { played_at: string; players: { display_name: string; is_bot: boolean; elo_before: number | null; elo_after: number | null }[] }[])]
    .sort((a, b) => new Date(a.played_at).getTime() - new Date(b.played_at).getTime());

  for (const game of sorted) {
    const me = game.players.find((p) => !p.is_bot && p.display_name === username);
    if (!me || me.elo_after == null) continue;
    const d = new Date(game.played_at);
    points.push({
      date: `${d.getMonth() + 1}/${d.getDate()}`,
      elo: me.elo_after,
    });
  }
  return points;
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
      <div className="p-8 text-sm text-text-secondary">{error}</div>
    );
  }

  if (!data) {
    return (
      <div className="p-8 flex items-center gap-2">
        <div className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  const { player, games } = data;
  const eloHistory = buildEloHistory(games as unknown[], username);

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
        <EloChart data={eloHistory} />
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
    <Suspense fallback={<div className="p-8" />}>
      <ProfileContent />
    </Suspense>
  );
}
