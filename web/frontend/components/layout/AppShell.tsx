"use client";

import { usePlayer } from "@/hooks/usePlayer";
import Sidebar from "./Sidebar";
import UsernameModal from "@/components/onboarding/UsernameModal";

export default function AppShell({ children }: { children: React.ReactNode }) {
  const { player, loaded, claim } = usePlayer();

  // Don't render until localStorage is read (avoids hydration flash)
  if (!loaded) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="w-6 h-6 border-2 border-accent border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar player={player} />
      <main className="flex-1 min-w-0">{children}</main>
      {!player && <UsernameModal onClaim={claim} />}
    </div>
  );
}
