"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect } from "react";
import type { Player } from "@/hooks/usePlayer";
import { useActiveGame } from "@/hooks/useActiveGame";
import { Gamepad2, Trophy, BookOpen, Settings, HelpCircle, ChevronLeft, ChevronRight } from "lucide-react";

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
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem("sidebar-collapsed");
    if (stored !== null) setCollapsed(stored === "true");
  }, []);

  function toggleCollapsed() {
    const next = !collapsed;
    setCollapsed(next);
    localStorage.setItem("sidebar-collapsed", String(next));
  }

  const expanded = !collapsed;

  return (
    <aside
      className={`shrink-0 flex-col border-r border-border bg-background transition-[width] duration-200 ease-out ${
        isGamePage ? "hidden lg:flex" : "flex"
      } ${expanded ? "w-12 lg:w-[200px]" : "w-12"}`}
      style={{ minHeight: "100dvh" }}
    >
      {/* Brand */}
      <div className="flex items-center gap-2 px-2 pt-5 pb-3 lg:px-3 lg:pt-8 lg:pb-5 overflow-hidden">
        <span className="text-lg shrink-0">🥚</span>
        {expanded && (
          <span className="hidden lg:inline text-sm font-bold text-foreground tracking-tight whitespace-nowrap">
            Chucking Eggs
          </span>
        )}
      </div>

      {/* Nav */}
      <nav className="flex flex-col gap-0.5 px-1 lg:px-2 flex-1">
        {navItems.map(({ label, href, icon: Icon, external }) => {
          const resolvedHref = label === "Play" && isInGame ? "/game" : href;
          const resolvedLabel = label === "Play" && isInGame ? "Resume" : label;
          const isActive =
            !external &&
            (pathname === resolvedHref || (label === "Play" && pathname === "/game"));
          const className = `flex items-center justify-center gap-2.5 px-2 py-3 lg:py-2 rounded-lg text-sm font-medium transition-all duration-150 ease-out border-l-2 ${
            expanded ? "lg:justify-start lg:px-3" : ""
          } ${
            isActive
              ? "border-accent text-accent bg-accent/5"
              : "border-transparent text-foreground hover:bg-accent/8 hover:text-accent"
          }`;
          if (external) {
            return (
              <a key={label} href={href} target="_blank" rel="noopener noreferrer" className={className}>
                <Icon size={15} strokeWidth={2} />
                {expanded && <span className="hidden lg:inline">{resolvedLabel}</span>}
              </a>
            );
          }
          return (
            <Link key={label} href={resolvedHref} className={className}>
              <Icon size={15} strokeWidth={isActive ? 2.5 : 2} />
              {expanded && <span className="hidden lg:inline">{resolvedLabel}</span>}
            </Link>
          );
        })}

        {/* Collapse toggle — desktop only */}
        <button
          onClick={toggleCollapsed}
          className={`hidden lg:flex items-center gap-2.5 px-2 py-2 mt-auto rounded-lg text-sm font-medium transition-all duration-150 ease-out border-l-2 border-transparent text-text-secondary hover:bg-accent/8 hover:text-accent ${
            expanded ? "justify-start px-3" : "justify-center"
          }`}
        >
          {expanded ? (
            <>
              <ChevronLeft size={15} strokeWidth={2} />
              <span>Collapse</span>
            </>
          ) : (
            <ChevronRight size={15} strokeWidth={2} />
          )}
        </button>
      </nav>

      {/* Player footer — avatar letter only, links to profile */}
      {player && (
        <div className="px-2 py-4 border-t border-border flex justify-center">
          <Link
            href={`/profile/${player.username}`}
            className="w-7 h-7 rounded-full bg-accent/10 text-accent flex items-center justify-center text-xs font-bold hover:opacity-70 transition-opacity duration-150"
          >
            {player.username[0].toUpperCase()}
          </Link>
        </div>
      )}
    </aside>
  );
}
