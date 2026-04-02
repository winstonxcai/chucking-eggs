"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { Player } from "@/hooks/usePlayer";
import { useActiveGame } from "@/hooks/useActiveGame";
import { Gamepad2, Trophy, BookOpen, Settings, HelpCircle } from "lucide-react";

interface SidebarProps {
  player: Player | null;
}

const navItems = [
  { label: "Play", href: "/", icon: Gamepad2 },
  { label: "Leaderboard", href: "/leaderboard", icon: Trophy },
  { label: "Rules", href: "https://www.pagat.com/climbing/guandan.html", icon: HelpCircle, external: true },
  { label: "Attributions", href: "/attributions", icon: BookOpen },
  { label: "Settings", href: "/settings", icon: Settings },
];

export default function Sidebar({ player }: SidebarProps) {
  const pathname = usePathname();
  const { isInGame } = useActiveGame();
  const isGamePage = pathname === "/game";

  return (
    <aside
      className={`shrink-0 flex-col border-r border-border bg-background ${
        isGamePage
          ? "hidden lg:flex lg:w-[200px]"
          : "flex w-12 lg:w-[200px]"
      }`}
      style={{ minHeight: "100dvh" }}
    >
      {/* Brand */}
      <div className="px-2 pt-5 pb-3 lg:px-5 lg:pt-8 lg:pb-5">
        <span className="hidden lg:inline text-lg font-bold text-foreground tracking-tight">Chucking Eggs</span>
        <span className="lg:hidden text-sm font-bold text-foreground tracking-tight">CE</span>
      </div>

      {/* Nav */}
      <nav className="flex flex-col gap-0.5 px-1 lg:px-3 flex-1">
        {navItems.map(({ label, href, icon: Icon, external }) => {
          const resolvedHref = label === "Play" && isInGame ? "/game" : href;
          const resolvedLabel = label === "Play" && isInGame ? "Resume" : label;
          const isActive = !external && (pathname === resolvedHref || (label === "Play" && pathname === "/game"));
          const className = `flex items-center justify-center lg:justify-start gap-2.5 px-2 lg:px-3 py-3 lg:py-2 rounded-lg text-sm font-medium transition-all duration-150 ease-out border-l-2 ${
            isActive
              ? "border-accent text-accent bg-accent/5"
              : "border-transparent text-foreground hover:bg-accent/8 hover:text-accent"
          }`;
          if (external) {
            return (
              <a key={label} href={href} target="_blank" rel="noopener noreferrer" className={className}>
                <Icon size={15} strokeWidth={2} />
                <span className="hidden lg:inline">{resolvedLabel}</span>
              </a>
            );
          }
          return (
            <Link key={label} href={resolvedHref} className={className}>
              <Icon size={15} strokeWidth={isActive ? 2.5 : 2} />
              <span className="hidden lg:inline">{resolvedLabel}</span>
            </Link>
          );
        })}
      </nav>

      {/* Player footer */}
      {player && (
        <div className="px-2 lg:px-4 py-4 border-t border-border">
          <Link
            href={`/profile/${player.username}`}
            className="flex items-center justify-center lg:justify-start gap-2.5 hover:opacity-70 transition-opacity duration-150"
          >
            <div className="w-7 h-7 rounded-full bg-accent/10 text-accent flex items-center justify-center text-xs font-bold shrink-0">
              {player.username[0].toUpperCase()}
            </div>
            <div className="hidden lg:flex flex-col gap-0 min-w-0">
              <span className="text-sm font-semibold text-foreground truncate">{player.username}</span>
              <span className="text-xs text-text-secondary">Elo {player.elo}</span>
            </div>
          </Link>
        </div>
      )}
    </aside>
  );
}
