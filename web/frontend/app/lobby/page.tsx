"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import type { RoomStatus, RoomMode } from "@/lib/types";
import { DIFFICULTY_INFO, OUR_BOTS, COMPETITION_BOTS } from "@/lib/bots";
import { STORAGE_KEYS } from "@/lib/storage-keys";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const SEAT_LABELS = ["South (You)", "West", "North", "East"] as const;

const MODE_LABEL: Record<RoomMode, string> = {
  solo: "Solo",
  duo: "2-Player",
  quad: "4-Player",
};


function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const copy = useCallback(async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [text]);
  return (
    <button
      onClick={copy}
      className="text-xs text-accent hover:underline transition-colors"
    >
      {copied ? "Copied!" : "Copy"}
    </button>
  );
}

function LobbyContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const gameId = searchParams.get("game_id");
  const seat = parseInt(searchParams.get("seat") || "0", 10);
  const token = searchParams.get("token");

  const [status, setStatus] = useState<RoomStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [difficulty, setDifficulty] = useState("medium");
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Persist session to survive page refresh
  useEffect(() => {
    if (gameId && token) {
      try {
        sessionStorage.setItem(STORAGE_KEYS.GAME_ID, gameId);
        sessionStorage.setItem(STORAGE_KEYS.RECONNECT_TOKEN, token);
        sessionStorage.setItem(STORAGE_KEYS.SEAT, String(seat));
      } catch { /* Safari private mode */ }
    }
  }, [gameId, seat, token]);

  const fetchStatus = useCallback(async () => {
    if (!gameId) return;
    try {
      const res = await fetch(`${API_BASE}/api/room/${gameId}/status`);
      if (!res.ok) {
        setError("Room not found");
        return;
      }
      const data: RoomStatus = await res.json();
      setStatus(data);

      // Navigate to game once all humans are connected and game has started
      if (data.started) {
        if (pollRef.current) clearInterval(pollRef.current);
        router.push(`/game?game_id=${gameId}&seat=${seat}&token=${encodeURIComponent(token || "")}`);
      }
    } catch {
      setError("Failed to reach server");
    }
  }, [gameId, seat, token, router]);

  useEffect(() => {
    if (!gameId || !token) {
      setError("Missing game ID or token");
      return;
    }
    fetchStatus();
    pollRef.current = setInterval(fetchStatus, 2000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [gameId, token, fetchStatus]);

  const changeDifficulty = useCallback(async (diff: string) => {
    setDifficulty(diff);
    try {
      await fetch(`${API_BASE}/api/room/${gameId}/set_difficulty`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ difficulty: diff }),
      });
    } catch { /* non-critical */ }
  }, [gameId]);

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="text-center flex flex-col items-center gap-4">
          <span className="text-text-secondary">{error}</span>
          <button
            className="text-sm text-accent hover:underline"
            onClick={() => router.push("/")}
          >
            Back to home
          </button>
        </div>
      </div>
    );
  }

  if (!status) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  const humanSeats = status.seats.filter((s) => s.is_human);
  const connectedCount = humanSeats.filter((s) => s.connected).length;
  const totalHumans = humanSeats.length;
  const isHost = seat === 0;
  const isDuo = status.mode === "duo";

  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center px-4">
      <div className="max-w-sm w-full flex flex-col gap-6">
        {/* Header */}
        <div className="flex flex-col items-center gap-1">
          <h1 className="text-2xl font-bold text-foreground">Waiting for players</h1>
          <span className="text-sm text-text-secondary">
            {MODE_LABEL[status.mode]} · {connectedCount}/{totalHumans} connected
          </span>
        </div>

        {/* Duo difficulty picker (host only) */}
        {isDuo && isHost && (
          <div className="bg-surface border border-border rounded-xl p-4 flex flex-col gap-2">
            <span className="text-xs font-semibold text-text-secondary tracking-widest uppercase">Bot Difficulty</span>
            <select
              value={difficulty}
              onChange={(e) => changeDifficulty(e.target.value)}
              className="w-full px-3 py-2 bg-background border border-border rounded-lg text-sm text-foreground focus:outline-none focus:border-accent transition-colors"
            >
              <optgroup label="Our Bots">
                {OUR_BOTS.map((diff) => {
                  const info = DIFFICULTY_INFO[diff];
                  return (
                    <option key={diff} value={diff}>
                      {info.emoji} {info.label} — {info.description}
                    </option>
                  );
                })}
              </optgroup>
              <optgroup label="Competition Bots">
                {COMPETITION_BOTS.map((diff) => {
                  const info = DIFFICULTY_INFO[diff];
                  return (
                    <option key={diff} value={diff}>
                      {info.emoji} {info.label} — {info.description}
                    </option>
                  );
                })}
              </optgroup>
            </select>
          </div>
        )}

        {/* Room code */}
        {status.room_code && (
          <div className="bg-surface border border-border rounded-xl p-5 flex flex-col items-center gap-3">
            <span className="text-xs font-semibold text-text-secondary tracking-widest uppercase">Room Code</span>
            <span data-testid="room-code" className="text-4xl font-bold tracking-widest text-foreground font-mono">
              {status.room_code}
            </span>
            <div className="flex items-center gap-2 text-xs text-text-secondary">
              <span>Share this code with friends</span>
              <CopyButton text={status.room_code} />
            </div>
          </div>
        )}

        {/* Seat list */}
        <div className="bg-surface border border-border rounded-xl overflow-hidden">
          {status.seats.map((s) => (
            <div
              key={s.seat}
              className="flex items-center justify-between px-4 py-3 border-b border-border last:border-0"
            >
              <div className="flex items-center gap-3">
                <div
                  className={`w-2 h-2 rounded-full ${
                    s.connected ? "bg-green-500" : s.is_human ? "bg-amber-400" : "bg-border"
                  }`}
                />
                <div className="flex flex-col">
                  <span className="text-sm font-medium text-foreground">
                    {s.seat === seat ? "You" : s.name}
                  </span>
                  <span className="text-xs text-text-secondary">
                    {SEAT_LABELS[s.seat]}
                  </span>
                </div>
              </div>
              <span className="text-xs text-text-secondary">
                {s.is_human
                  ? s.connected
                    ? "Connected"
                    : "Waiting..."
                  : "AI"}
              </span>
            </div>
          ))}
        </div>

        {/* Status indicator */}
        <div className="flex items-center justify-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
          <span className="text-xs text-text-secondary">
            Waiting for all players to join…
          </span>
        </div>

        <button
          className="text-xs text-text-secondary hover:text-foreground transition-colors"
          onClick={() => router.push("/")}
        >
          Cancel and go home
        </button>
      </div>
    </div>
  );
}

export default function LobbyPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen flex items-center justify-center bg-background">
          <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
        </div>
      }
    >
      <LobbyContent />
    </Suspense>
  );
}
