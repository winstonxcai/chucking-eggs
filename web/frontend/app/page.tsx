"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { DIFFICULTY_INFO } from "@/lib/bots";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Ordered by calibrated Glicko-2 ELO (see runs/wr_matrix_v2/)
const difficulties = ["wjsd", "liuzha", "hulalala", "easy", "medium", "competition", "casual", "hard", "master", "yaoji", "jidan", "expert"] as const;

export default function Home() {
  const router = useRouter();
  const [creatingRoom, setCreatingRoom] = useState(false);

  async function handleCreateRoom(difficulty: string) {
    setCreatingRoom(true);
    try {
      const res = await fetch(`${API_BASE}/api/room/create`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: "duo", difficulty }),
      });
      const data = await res.json();
      router.push(
        `/lobby?game_id=${data.game_id}&seat=${data.seat}&token=${encodeURIComponent(data.reconnect_token)}`
      );
    } catch {
      setCreatingRoom(false);
    }
  }

  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center px-4">
      <div className="max-w-md w-full flex flex-col items-center gap-8">
        {/* Title */}
        <div className="flex flex-col items-center gap-2">
          <h1 className="text-4xl font-bold text-foreground tracking-tight">
            掼蛋
          </h1>
          <h2 className="text-xl font-medium text-text-secondary">Guan Dan</h2>
          <p className="text-sm text-text-secondary text-center mt-2">
            Play the classic Chinese card game against AI opponents
          </p>
        </div>

        {/* Solo difficulty picker */}
        <div className="w-full flex flex-col gap-3">
          <span className="text-xs font-semibold text-text-secondary tracking-wider uppercase text-center">
            Play Solo
          </span>
          <div className="grid grid-cols-2 gap-3">
            {difficulties.map((diff) => {
              const info = DIFFICULTY_INFO[diff];
              return (
                <button
                  key={diff}
                  className="flex flex-col items-center gap-2 p-5 bg-surface border border-border rounded-xl hover:border-accent hover:shadow-md transition-all cursor-pointer group"
                  onClick={() => router.push(`/game?difficulty=${diff}`)}
                >
                  <span className="text-3xl">{info.emoji}</span>
                  <span className="text-sm font-semibold text-foreground group-hover:text-accent transition-colors">
                    {info.label}
                  </span>
                  <span className="text-xs text-text-secondary text-center">
                    {info.description}
                  </span>
                  <span className="text-xs font-mono text-text-secondary opacity-60">
                    {info.elo} ELO
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Multiplayer section */}
        <div className="w-full flex flex-col gap-3">
          <span className="text-xs font-semibold text-text-secondary tracking-wider uppercase text-center">
            Play with Friends
          </span>
          <div className="flex gap-3">
            <button
              className="flex-1 py-3 bg-surface border border-border rounded-xl text-sm font-semibold text-foreground hover:border-accent hover:text-accent transition-all disabled:opacity-50"
              onClick={() => handleCreateRoom("medium")}
              disabled={creatingRoom}
            >
              {creatingRoom ? "Creating…" : "Create Room"}
            </button>
            <button
              className="flex-1 py-3 bg-surface border border-border rounded-xl text-sm font-semibold text-foreground hover:border-accent hover:text-accent transition-all"
              onClick={() => router.push("/join")}
            >
              Join Room
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
