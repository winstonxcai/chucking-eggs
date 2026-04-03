"use client";

import { useEffect, useState } from "react";

export default function LandscapeGate({ children }: { children: React.ReactNode }) {
  const [showRotate, setShowRotate] = useState(false);

  useEffect(() => {
    const isTouch = navigator.maxTouchPoints > 0;
    if (!isTouch) return;
    const mq = window.matchMedia("(orientation: portrait)");
    const update = (e: MediaQueryListEvent | MediaQueryList) => setShowRotate(e.matches);
    update(mq);
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);

  return (
    <>
      {children}
      {showRotate && (
        <div className="fixed inset-0 z-[9999] bg-background flex flex-col items-center justify-center gap-5 px-8">
          <svg
            className="w-16 h-16 text-accent"
            viewBox="0 0 64 64"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            {/* Phone outline rotated to portrait */}
            <rect x="20" y="8" width="24" height="40" rx="4" />
            <circle cx="32" cy="44" r="1.5" fill="currentColor" stroke="none" />
            {/* Rotation arrow */}
            <path d="M10 32 C10 18 18 10 32 10" strokeDasharray="4 3" />
            <polyline points="29 6 32 10 35 6" />
          </svg>
          <div className="flex flex-col items-center gap-1.5 text-center">
            <span className="text-base font-semibold text-foreground">Rotate your device</span>
            <span className="text-sm text-text-secondary">
              Guan Dan is best played in landscape mode
            </span>
          </div>
        </div>
      )}
    </>
  );
}
