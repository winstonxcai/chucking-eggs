"use client";

import { useCallback, useState } from "react";
import { usePlayer } from "@/hooks/usePlayer";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function SettingsPage() {
  const { player } = usePlayer();
  const [editingEmail, setEditingEmail] = useState(false);
  const [email, setEmail] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSaveEmail = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    if (!player) return;
    setSaving(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/auth/update-email`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Player-ID": player.playerId,
        },
        body: JSON.stringify({ email }),
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Failed to update email");
      }
      setSaved(true);
      setEditingEmail(false);
      setTimeout(() => setSaved(false), 3000);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to save");
    } finally {
      setSaving(false);
    }
  }, [player, email]);

  return (
    <div className="max-w-2xl mx-auto px-8 py-8 flex flex-col gap-6">
      <h1 className="text-2xl font-bold text-foreground">Settings</h1>

      {/* Account section */}
      <div className="bg-surface border border-border rounded-xl overflow-hidden">
        <div className="px-5 py-3 border-b border-border">
          <span className="text-xs font-semibold text-text-secondary uppercase tracking-wide">Account</span>
        </div>

        {/* Username row */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-border">
          <div className="flex flex-col gap-0.5">
            <span className="text-xs text-text-secondary">Username</span>
            <span className="text-sm font-medium text-foreground">{player?.username ?? "—"}</span>
          </div>
          <span className="text-xs text-text-secondary border border-border rounded-full px-2.5 py-0.5">locked</span>
        </div>

        {/* Email row */}
        <div className="flex flex-col px-5 py-4">
          <div className="flex items-center justify-between">
            <div className="flex flex-col gap-0.5">
              <span className="text-xs text-text-secondary">Email</span>
              <span className="text-sm text-foreground">{editingEmail ? "" : "Add for account recovery"}</span>
            </div>
            {!editingEmail && (
              <button
                className="text-sm text-accent hover:underline"
                onClick={() => setEditingEmail(true)}
              >
                Edit
              </button>
            )}
          </div>

          {editingEmail && (
            <form onSubmit={handleSaveEmail} className="flex flex-col gap-3 mt-3">
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="your@email.com"
                required
                className="w-full px-3 py-2 rounded-lg border border-border text-sm text-foreground bg-background placeholder:text-text-secondary outline-none focus:ring-2 focus:ring-accent/30 focus:border-accent transition-all"
              />
              {error && <span className="text-xs text-team-red">{error}</span>}
              <div className="flex gap-2">
                <button
                  type="submit"
                  disabled={saving}
                  className="px-4 py-2 bg-accent text-white text-sm font-semibold rounded-lg hover:bg-accent-hover transition-colors disabled:opacity-50"
                >
                  {saving ? "Saving…" : "Save"}
                </button>
                <button
                  type="button"
                  className="px-4 py-2 text-sm text-text-secondary hover:text-foreground transition-colors"
                  onClick={() => setEditingEmail(false)}
                >
                  Cancel
                </button>
              </div>
            </form>
          )}

          {saved && <span className="text-xs text-team-green mt-2">Email saved.</span>}

          <span className="text-xs text-text-secondary mt-2">
            Add your email to recover your account from any device.
          </span>
        </div>
      </div>

      {/* Future settings placeholder */}
      <div className="bg-surface border border-border rounded-xl overflow-hidden opacity-40 select-none">
        <div className="px-5 py-3 border-b border-border">
          <span className="text-xs font-semibold text-text-secondary uppercase tracking-wide">Preferences</span>
        </div>
        <div className="px-5 py-4 text-sm text-text-secondary">Sound, theme, language — coming soon</div>
      </div>
    </div>
  );
}
