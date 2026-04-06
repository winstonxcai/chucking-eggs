"use client";

import { useEffect, useState } from "react";
import type { CardDTO } from "@/lib/types";
import CardComponent from "./CardComponent";

interface FlyingCardsProps {
  cards: CardDTO[];
  fromRects: DOMRect[];
  toRect: DOMRect;
  compact?: boolean;
  onArrived?: () => void;
}

export default function FlyingCards({ cards, fromRects, toRect, compact, onArrived }: FlyingCardsProps) {
  const [arrived, setArrived] = useState(false);

  useEffect(() => {
    // Double rAF to ensure initial position is painted before triggering transition
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        setArrived(true);
        onArrived?.();
      });
    });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Use actual rendered card dimensions from the source rects
  const cardW = fromRects[0].width;
  const cardH = fromRects[0].height;
  const totalWidth = cards.length * cardW + (cards.length - 1) * 2;
  const targetBaseX = toRect.left + toRect.width / 2 - totalWidth / 2;
  const targetY = toRect.bottom - cardH;

  return (
    <div className="fixed inset-0 z-50 pointer-events-none">
      {cards.map((card, i) => {
        const from = fromRects[i];
        const targetX = targetBaseX + i * (cardW + 2);

        const dx = from.left - targetX;
        const dy = from.top - targetY;

        const style: React.CSSProperties = {
          left: targetX,
          top: targetY,
          width: cardW,
          height: cardH,
          transform: arrived ? "translate(0, 0)" : `translate(${dx}px, ${dy}px)`,
          transition: arrived ? "transform 300ms cubic-bezier(0.4, 0, 0.2, 1)" : "none",
        };

        return (
          <div key={card.id} className="absolute" style={style}>
            <CardComponent card={card} size={compact ? "xs" : "sm"} />
          </div>
        );
      })}
    </div>
  );
}
