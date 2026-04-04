"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { DIFFICULTY_INFO, OUR_BOTS, COMPETITION_BOTS } from "@/lib/bots";
import { useActiveGame } from "@/hooks/useActiveGame";
import { usePlayer } from "@/hooks/usePlayer";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const router = useRouter();
  const [creatingRoom, setCreatingRoom] = useState<"duo" | "quad" | null>(null);
  const [soloOpen, setSoloOpen] = useState(false);
  const { isInGame, getResumeParams, clearGame } = useActiveGame();
  const { player } = usePlayer();
  const [forfeitConfirm, setForfeitConfirm] = useState(false);
  const [forfeiting, setForfeiting] = useState(false);

  async function handleCreateRoom(mode: "duo" | "quad", difficulty: string) {
    setCreatingRoom(mode);
    try {
      const res = await fetch(`${API_BASE}/api/room/create`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode, difficulty }),
      });
      const data = await res.json();
      setCreatingRoom(null);
      router.push(
        `/lobby?game_id=${data.game_id}&seat=${data.seat}&token=${encodeURIComponent(data.reconnect_token)}`
      );
    } catch {
      setCreatingRoom(null);
    }
  }

  return (
    <div className="min-h-screen flex flex-col items-center pt-[12vh] sm:pt-0 sm:justify-center px-4">
      <div className="max-w-md w-full flex flex-col items-center gap-8">
        {/* Title — hidden on mobile to save space */}
        <div className="hidden sm:flex flex-col items-center gap-1">
          <h1 className="text-4xl font-bold text-foreground tracking-tight">Guan Dan</h1>
          <span className="text-sm text-text-secondary font-normal tracking-widest">掼蛋</span>
          <p className="text-sm text-text-secondary text-center mt-2">
            Play the classic Chinese card game against AI opponents
          </p>
        </div>

        {/* Active multiplayer game banner */}
        {isInGame && (
          <div className="w-full bg-surface border border-accent/30 rounded-xl p-4 flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-accent animate-pulse" />
              <span className="text-sm font-semibold text-foreground">You have an active game</span>
            </div>
            <div className="flex gap-3">
              <button
                className="flex-1 py-2.5 bg-accent text-white font-semibold rounded-xl hover:bg-accent-hover transition-colors"
                onClick={() => {
                  const params = getResumeParams();
                  if (params) {
                    router.push(`/game?game_id=${params.gameId}&seat=${params.seat}&token=${encodeURIComponent(params.token)}`);
                  }
                }}
              >
                Rejoin
              </button>
              {!forfeitConfirm ? (
                <button
                  className="flex-1 py-2.5 bg-background border border-border rounded-xl text-sm font-semibold text-text-secondary hover:border-team-red hover:text-team-red transition-colors"
                  onClick={() => setForfeitConfirm(true)}
                >
                  Forfeit
                </button>
              ) : (
                <div className="flex-1 flex flex-col gap-1.5">
                  <span className="text-xs text-team-red text-center font-medium">Forfeit? You will lose ELO.</span>
                  <div className="flex gap-2">
                    <button
                      className="flex-1 py-1.5 bg-team-red text-white text-xs font-semibold rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50"
                      disabled={forfeiting}
                      onClick={async () => {
                        const params = getResumeParams();
                        if (!params || !player) return;
                        setForfeiting(true);
                        try {
                          await fetch(`${API_BASE}/api/room/${params.gameId}/forfeit`, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ player_id: player.playerId }),
                          });
                        } catch { /* best effort */ }
                        clearGame();
                        setForfeitConfirm(false);
                        setForfeiting(false);
                      }}
                    >
                      {forfeiting ? "..." : "Confirm"}
                    </button>
                    <button
                      className="flex-1 py-1.5 bg-background border border-border text-xs font-medium rounded-lg hover:border-foreground transition-colors"
                      onClick={() => setForfeitConfirm(false)}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Solo difficulty picker */}
        <div className="w-full flex flex-col gap-1">
          <button
            className="w-full flex items-center justify-between px-4 py-3 bg-accent text-white rounded-xl hover:bg-accent-hover transition-all duration-150 ease-out group"
            onClick={() => setSoloOpen((o) => !o)}
          >
            <span className="text-sm font-semibold text-white">
              Play Solo
            </span>
            <svg
              className={`w-4 h-4 text-white/70 transition-transform duration-200 ease-out ${soloOpen ? "rotate-180" : ""}`}
              viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
            >
              <polyline points="4 6 8 10 12 6" />
            </svg>
          </button>

          <div className={`overflow-hidden transition-all duration-200 ease-out ${soloOpen ? "max-h-[900px] opacity-100" : "max-h-0 opacity-0"}`}>
            <div className="border border-border rounded-xl overflow-hidden mt-1">
              {/* Our bots */}
              <div className="px-4 py-1.5 bg-surface border-b border-border">
                <span className="text-[10px] font-semibold text-text-secondary tracking-widest uppercase">Our Bots</span>
              </div>
              {OUR_BOTS.map((diff, i) => {
                const info = DIFFICULTY_INFO[diff];
                return (
                  <button
                    key={diff}
                    className={`w-full flex items-center gap-3 px-4 py-3 hover:bg-background transition-colors duration-150 text-left group ${
                      i < OUR_BOTS.length - 1 ? "border-b border-border" : ""
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
              {/* Competition bots */}
              <div className="px-4 py-1.5 bg-surface border-t border-b border-border">
                <span className="text-[10px] font-semibold text-text-secondary tracking-widest uppercase">Competition Bots</span>
              </div>
              {COMPETITION_BOTS.map((diff, i) => {
                const info = DIFFICULTY_INFO[diff];
                return (
                  <button
                    key={diff}
                    className={`w-full flex items-center gap-3 px-4 py-3 hover:bg-background transition-colors duration-150 text-left group ${
                      i < COMPETITION_BOTS.length - 1 ? "border-b border-border" : ""
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
              onClick={() => handleCreateRoom("duo", "easy")}
              disabled={creatingRoom !== null || isInGame}
            >
              {creatingRoom === "duo" ? "Creating…" : "2-Player"}
            </button>
            <button
              className="flex-1 py-3 bg-background border border-border rounded-xl text-sm font-semibold text-foreground hover:border-accent hover:text-accent hover:bg-accent/5 transition-all duration-150 ease-out disabled:opacity-50"
              onClick={() => handleCreateRoom("quad", "easy")}
              disabled={creatingRoom !== null || isInGame}
            >
              {creatingRoom === "quad" ? "Creating…" : "4-Player"}
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
