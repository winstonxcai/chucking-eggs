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
  const [soloOpen, setSoloOpen] = useState(false);

  async function handleCreateRoom(mode: "duo" | "quad", difficulty: string) {
    setCreatingRoom(true);
    try {
      const res = await fetch(`${API_BASE}/api/room/create`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode, difficulty }),
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
    <div className="flex-1 flex flex-col items-center justify-center px-4 py-12">
      <div className="max-w-md w-full flex flex-col items-center gap-8">
        {/* Title */}
        <div className="flex flex-col items-center gap-1">
          <h1 className="text-4xl font-bold text-foreground tracking-tight">Guan Dan</h1>
          <span className="text-sm text-text-secondary font-normal tracking-widest">掼蛋</span>
          <p className="text-sm text-text-secondary text-center mt-2">
            Play the classic Chinese card game against AI opponents
          </p>
        </div>

        {/* Solo difficulty picker */}
        <div className="w-full flex flex-col gap-1">
          <button
            className="w-full flex items-center justify-between px-4 py-3 bg-surface border border-border rounded-xl hover:border-accent transition-all duration-150 ease-out group"
            onClick={() => setSoloOpen((o) => !o)}
          >
            <span className="text-sm font-semibold text-foreground group-hover:text-accent transition-colors duration-150">
              Play Solo
            </span>
            <svg
              className={`w-4 h-4 text-text-secondary transition-transform duration-200 ease-out ${soloOpen ? "rotate-180" : ""}`}
              viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
            >
              <polyline points="4 6 8 10 12 6" />
            </svg>
          </button>

          <div className={`overflow-hidden transition-all duration-200 ease-out ${soloOpen ? "max-h-[700px] opacity-100" : "max-h-0 opacity-0"}`}>
            <div className="border border-border rounded-xl overflow-hidden mt-1">
              {difficulties.map((diff, i) => {
                const info = DIFFICULTY_INFO[diff];
                return (
                  <button
                    key={diff}
                    className={`w-full flex items-center gap-3 px-4 py-3 hover:bg-background transition-colors duration-150 text-left group ${
                      i < difficulties.length - 1 ? "border-b border-border" : ""
                    }`}
                    onClick={() => router.push(`/game?difficulty=${diff}`)}
                  >
                    <span className="text-xl w-7 shrink-0">{info.emoji}</span>
                    <span className="text-sm font-semibold text-foreground group-hover:text-accent transition-colors duration-150 w-24 shrink-0">
                      {info.label}
                    </span>
                    <span className="text-xs text-text-secondary flex-1 truncate">{info.description}</span>
                    <span className="text-xs font-mono text-text-secondary opacity-60 shrink-0">{info.elo}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Multiplayer section */}
        <div className="w-full bg-surface border border-border rounded-xl p-4 flex flex-col gap-4">
          <span className="text-xs font-semibold text-text-secondary tracking-wider uppercase text-center">
            Play with Friends
          </span>
          <div className="flex gap-3">
            <button
              className="flex-1 py-3 bg-background border border-border rounded-xl text-sm font-semibold text-foreground hover:border-accent hover:text-accent hover:bg-accent/5 transition-all duration-150 ease-out disabled:opacity-50"
              onClick={() => handleCreateRoom("duo", "medium")}
              disabled={creatingRoom}
            >
              {creatingRoom ? "Creating…" : "2-Player"}
            </button>
            <button
              className="flex-1 py-3 bg-background border border-border rounded-xl text-sm font-semibold text-foreground hover:border-accent hover:text-accent hover:bg-accent/5 transition-all duration-150 ease-out disabled:opacity-50"
              onClick={() => handleCreateRoom("quad", "medium")}
              disabled={creatingRoom}
            >
              {creatingRoom ? "Creating…" : "4-Player"}
            </button>
            <button
              className="flex-1 py-3 bg-background border border-border rounded-xl text-sm font-semibold text-foreground hover:border-accent hover:text-accent hover:bg-accent/5 transition-all duration-150 ease-out"
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
