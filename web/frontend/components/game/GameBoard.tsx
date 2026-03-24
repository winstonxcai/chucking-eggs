"use client";

import { useCallback, useMemo, useState } from "react";
import type { CardDTO, ComboDTO, GameOverMsg, GameState } from "@/lib/types";
import { findMatchingCombo } from "@/lib/cards";
import PlayerHand from "./PlayerHand";
import OpponentPanel from "./OpponentPanel";
import PlayArea from "./PlayArea";
import GameControls from "./GameControls";
import ComboBrowser from "./ComboBrowser";
import HandToolbar from "./HandToolbar";
import GameOverModal from "./GameOverModal";

interface GameBoardProps {
  gameState: GameState;
  aiThinking: number | null;
  gameOver: GameOverMsg | null;
  onPlayCards: (cardIds: string[]) => void;
  onPass: () => void;
  onPlayAgain: () => void;
}

export default function GameBoard({
  gameState,
  aiThinking,
  gameOver,
  onPlayCards,
  onPass,
  onPlayAgain,
}: GameBoardProps) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  const toggleCard = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const matchingCombo = useMemo(
    () => findMatchingCombo(selectedIds, gameState.legal_moves),
    [selectedIds, gameState.legal_moves]
  );

  const handlePlay = useCallback(() => {
    if (!matchingCombo) return;
    const ids = matchingCombo.cards.map((c) => c.id);
    onPlayCards(ids);
    setSelectedIds(new Set());
  }, [matchingCombo, onPlayCards]);

  const handlePass = useCallback(() => {
    onPass();
    setSelectedIds(new Set());
  }, [onPass]);

  const handleSelectCombo = useCallback((combo: ComboDTO) => {
    setSelectedIds(new Set(combo.cards.map((c) => c.id)));
  }, []);

  const handleFlushFind = useCallback(
    (suit: number) => {
      // Find all cards of this suit that could form a straight flush
      const suitCards = gameState.my_hand
        .filter((c) => c.suit === suit && c.rank >= 2 && c.rank <= 14)
        .sort((a, b) => a.rank - b.rank);

      // Find longest consecutive run of 5+
      if (suitCards.length < 5) return;

      let bestRun: CardDTO[] = [];
      let currentRun: CardDTO[] = [suitCards[0]];

      for (let i = 1; i < suitCards.length; i++) {
        if (suitCards[i].rank === suitCards[i - 1].rank + 1) {
          currentRun.push(suitCards[i]);
        } else if (suitCards[i].rank !== suitCards[i - 1].rank) {
          if (currentRun.length > bestRun.length) bestRun = currentRun;
          currentRun = [suitCards[i]];
        }
      }
      if (currentRun.length > bestRun.length) bestRun = currentRun;

      if (bestRun.length >= 5) {
        setSelectedIds(new Set(bestRun.slice(0, 5).map((c) => c.id)));
      }
    },
    [gameState.my_hand]
  );

  // Player layout: seat 0 = human (bottom), seat 2 = partner (top),
  // seat 1 = left opponent, seat 3 = right opponent
  const partner = gameState.players.find((p) => p.seat === 2);
  const leftOpp = gameState.players.find((p) => p.seat === 1);
  const rightOpp = gameState.players.find((p) => p.seat === 3);

  return (
    <div className="flex h-screen bg-background">
      {/* Main board area */}
      <div className="flex-1 flex flex-col p-6 gap-0">
        {/* Partner (top center) */}
        <div className="flex justify-center pb-3">
          {partner && (
            <OpponentPanel player={partner} thinking={aiThinking === 2} />
          )}
        </div>

        {/* Middle row: left opp | trick area | right opp */}
        <div className="flex-1 flex items-center justify-center gap-6">
          {leftOpp && (
            <OpponentPanel player={leftOpp} thinking={aiThinking === 1} />
          )}
          <PlayArea
            trick={gameState.current_trick}
            isLeading={gameState.is_leading}
            isMyTurn={gameState.is_my_turn}
            players={gameState.players}
          />
          {rightOpp && (
            <OpponentPanel player={rightOpp} thinking={aiThinking === 3} />
          )}
        </div>

        {/* Controls */}
        <div className="py-3">
          <GameControls
            matchingCombo={matchingCombo}
            isLeading={gameState.is_leading}
            isMyTurn={gameState.is_my_turn}
            onPlay={handlePlay}
            onPass={handlePass}
          />
        </div>

        {/* Player hand */}
        <div className="py-1">
          <PlayerHand
            cards={gameState.my_hand}
            selectedIds={selectedIds}
            onToggleCard={toggleCard}
          />
        </div>

        {/* Hand toolbar */}
        <div className="py-2">
          <HandToolbar onFlushFind={handleFlushFind} />
        </div>
      </div>

      {/* Sidebar: combo browser */}
      {gameState.is_my_turn && gameState.legal_moves.length > 0 && (
        <div className="w-[280px] bg-surface border-l border-border p-5 overflow-y-auto flex flex-col gap-4">
          <span className="text-[13px] font-semibold text-text-secondary tracking-wider uppercase">
            Legal Combos
          </span>
          <ComboBrowser
            legalMoves={gameState.legal_moves}
            onSelectCombo={handleSelectCombo}
          />
          <div className="mt-auto pt-4">
            <div className="p-3 bg-background rounded-lg">
              <span className="text-xs text-text-secondary">
                Click a combo to auto-select the cards in your hand
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Game over modal */}
      {gameOver && (
        <GameOverModal
          data={gameOver}
          humanSeat={gameState.my_seat}
          onPlayAgain={onPlayAgain}
        />
      )}
    </div>
  );
}
