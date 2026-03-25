"use client";

import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function JoinPage() {
  const router = useRouter();
  const [code, setCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleJoin = useCallback(async () => {
    const trimmed = code.trim().toUpperCase();
    if (trimmed.length !== 6) {
      setError("Room code must be 6 characters");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/room/join/${trimmed}`, {
        method: "POST",
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        setError(data.detail || "Room not found or already full");
        return;
      }
      const data = await res.json();
      router.push(
        `/lobby?game_id=${data.game_id}&seat=${data.seat}&token=${encodeURIComponent(data.reconnect_token)}`
      );
    } catch {
      setError("Could not reach server");
    } finally {
      setLoading(false);
    }
  }, [code, router]);

  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center px-4">
      <div className="max-w-sm w-full flex flex-col gap-6">
        <div className="flex flex-col items-center gap-1">
          <h1 className="text-2xl font-bold text-foreground">Join a game</h1>
          <p className="text-sm text-text-secondary">Enter the 6-character room code</p>
        </div>

        <div className="flex flex-col gap-3">
          <input
            className="w-full text-center text-2xl font-mono tracking-widest uppercase px-4 py-4 bg-surface border border-border rounded-xl focus:outline-none focus:border-accent transition-colors"
            maxLength={6}
            placeholder="XXXXXX"
            value={code}
            onChange={(e) => setCode(e.target.value.toUpperCase().replace(/[^A-Z0-9]/g, ""))}
            onKeyDown={(e) => e.key === "Enter" && handleJoin()}
            autoFocus
          />
          {error && (
            <span className="text-xs text-red-500 text-center">{error}</span>
          )}
          <button
            className="w-full py-3 bg-accent text-white font-semibold rounded-xl hover:opacity-90 transition-opacity disabled:opacity-50"
            onClick={handleJoin}
            disabled={loading || code.length !== 6}
          >
            {loading ? "Joining…" : "Join Game"}
          </button>
        </div>

        <button
          className="text-xs text-text-secondary hover:text-foreground transition-colors text-center"
          onClick={() => router.push("/")}
        >
          ← Back
        </button>
      </div>
    </div>
  );
}
