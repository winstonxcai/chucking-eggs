"use client";

import { useCallback, useState } from "react";

interface UsernameModalProps {
  onClaim: (username: string, email?: string) => Promise<void>;
}

export default function UsernameModal({ onClaim }: UsernameModalProps) {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await onClaim(username.trim(), email.trim() || undefined);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }, [username, email, onClaim]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="bg-surface rounded-2xl shadow-2xl w-full max-w-sm mx-4 p-8 flex flex-col gap-6">
        <div className="flex flex-col gap-1">
          <h2 className="text-xl font-bold text-foreground tracking-tight">Welcome to Chucking Eggs</h2>
          <p className="text-sm text-text-secondary">Play Guan Dan against ranked bots and friends</p>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-semibold text-text-secondary tracking-wide uppercase" htmlFor="username">
              Username
            </label>
            <input
              id="username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="e.g. tiger_slayer"
              maxLength={20}
              required
              className={`w-full px-3 py-2.5 rounded-lg border text-sm text-foreground bg-background placeholder:text-text-secondary outline-none focus:ring-2 focus:ring-accent/30 transition-all ${
                error ? "border-team-red" : "border-border focus:border-accent"
              }`}
            />
            {error && <span className="text-xs text-team-red">{error}</span>}
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-semibold text-text-secondary tracking-wide uppercase" htmlFor="email">
              Email <span className="font-normal normal-case">(optional)</span>
            </label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="your@email.com"
              className="w-full px-3 py-2.5 rounded-lg border border-border text-sm text-foreground bg-background placeholder:text-text-secondary outline-none focus:ring-2 focus:ring-accent/30 focus:border-accent transition-all"
            />
            <span className="text-xs text-text-secondary">Add your email to recover your account from any device</span>
          </div>

          <button
            type="submit"
            disabled={loading || !username.trim()}
            className="w-full py-3 rounded-lg bg-accent text-white font-semibold text-sm hover:bg-accent-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed mt-1"
          >
            {loading ? "Claiming…" : "Start Playing"}
          </button>
        </form>
      </div>
    </div>
  );
}
