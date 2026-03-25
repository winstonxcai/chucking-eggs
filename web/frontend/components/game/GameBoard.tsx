"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CardDTO, CardGroup, ComboDTO, GameOverMsg, GameState, TrickAction } from "@/lib/types";
import { findMatchingCombo, validateCombo } from "@/lib/cards";
import PlayerHand from "./PlayerHand";
import OpponentPanel from "./OpponentPanel";
import GameControls from "./GameControls";
import ComboBrowser from "./ComboBrowser";
import HandToolbar from "./HandToolbar";
import GameOverModal from "./GameOverModal";
import CardComponent from "./CardComponent";
import FlyingCards from "./FlyingCards";
import type { ConnectionStatus } from "@/hooks/useGameSocket";

interface GameBoardProps {
  gameState: GameState;
  aiThinking: number | null;
  gameOver: GameOverMsg | null;
  connectionStatus?: ConnectionStatus;
  onPlayCards: (cardIds: string[]) => void;
  onPass: () => void;
  onPlayAgain: () => void;
  onCreateGroup: (cardIds: string[], comboType: string, comboName: string) => void;
  onDeleteGroup: (groupId: string) => void;
}

/** Render a single trick action (cards or "Pass") */
function TrickActionDisplay({ action }: { action: TrickAction | null }) {
  if (!action) return null;
  if (action.type === "pass") {
    return <span className="text-xs text-text-secondary italic">Pass</span>;
  }
  if (action.combo) {
    return (
      <div className="flex gap-0.5">
        {action.combo.cards.map((card) => (
          <CardComponent key={card.id} card={card} size="sm" />
        ))}
      </div>
    );
  }
  return null;
}

export default function GameBoard({
  gameState,
  aiThinking,
  gameOver,
  connectionStatus,
  onPlayCards,
  onPass,
  onPlayAgain,
  onCreateGroup,
  onDeleteGroup,
}: GameBoardProps) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [flyingCards, setFlyingCards] = useState<{
    cards: CardDTO[];
    fromRects: DOMRect[];
    toRect: DOMRect;
  } | null>(null);
  const tableSeat0Ref = useRef<HTMLDivElement>(null);
  const flyingStartRef = useRef<number | null>(null);

  // Optimistic local groups — updated immediately on Group/Ungroup, synced from server on game_state
  const [localGroups, setLocalGroups] = useState<CardGroup[]>(gameState.groups);
  useEffect(() => {
    setLocalGroups(gameState.groups);
  }, [gameState.groups]);

  const groupedCardIds = useMemo(
    () => new Set(localGroups.flatMap((g) => g.cardIds)),
    [localGroups]
  );

  // --- Card selection ---
  const toggleCard = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const matchingCombo = useMemo(
    () => findMatchingCombo(selectedIds, gameState.legal_moves, gameState.my_hand),
    [selectedIds, gameState.legal_moves, gameState.my_hand]
  );

  const handlePlay = useCallback(() => {
    if (!matchingCombo) return;
    const ids = matchingCombo.cards.map((c) => c.id);

    // Capture positions for fly animation
    const fromRects = ids.map((id) => {
      const el = document.querySelector(`[data-card-id="${id}"]`);
      return el?.getBoundingClientRect() ?? new DOMRect();
    });
    const toRect = tableSeat0Ref.current?.getBoundingClientRect() ?? new DOMRect();

    flyingStartRef.current = Date.now();
    setFlyingCards({ cards: matchingCombo.cards, fromRects, toRect });
    onPlayCards(ids);
    setSelectedIds(new Set());
  }, [matchingCombo, onPlayCards]);

  // Clear flying overlay once game state updates (server confirmed the play),
  // but always wait at least 300ms so the animation can complete.
  useEffect(() => {
    if (!flyingCards || flyingStartRef.current === null) return;
    const elapsed = Date.now() - flyingStartRef.current;
    const remaining = Math.max(0, 300 - elapsed);
    const t = setTimeout(() => {
      setFlyingCards(null);
      flyingStartRef.current = null;
    }, remaining);
    return () => clearTimeout(t);
  }, [gameState]); // eslint-disable-line react-hooks/exhaustive-deps

  const handlePass = useCallback(() => {
    onPass();
    setSelectedIds(new Set());
  }, [onPass]);

  const handleSelectCombo = useCallback((combo: ComboDTO) => {
    setSelectedIds(new Set(combo.cards.map((c) => c.id)));
  }, []);

  // --- Grouping (optimistic: update locally then sync to backend) ---
  const handleGroup = useCallback(() => {
    if (selectedIds.size === 0) return;
    const selectedCards = gameState.my_hand.filter((c) => selectedIds.has(c.id));
    const result = validateCombo(selectedCards);
    if (!result) return;
    const cardIds = selectedCards.map((c) => c.id);
    const idSet = new Set(cardIds);
    const newGroup: CardGroup = {
      id: `grp-temp-${Date.now()}`,
      cardIds,
      comboType: result.type,
      comboName: result.name,
    };
    setLocalGroups((prev) => [
      ...prev.filter((g) => !g.cardIds.some((cid) => idSet.has(cid))),
      newGroup,
    ]);
    onCreateGroup(cardIds, result.type, result.name);
    setSelectedIds(new Set());
  }, [selectedIds, gameState.my_hand, onCreateGroup]);

  const handleUngroup = useCallback(() => {
    if (selectedIds.size === 0) return;
    const toDelete = localGroups.find((g) => g.cardIds.some((cid) => selectedIds.has(cid)));
    if (toDelete) {
      setLocalGroups((prev) => prev.filter((g) => g.id !== toDelete.id));
      onDeleteGroup(toDelete.id);
    }
    setSelectedIds(new Set());
  }, [selectedIds, localGroups, onDeleteGroup]);

  const handleGroupClick = useCallback((group: CardGroup) => {
    setSelectedIds((prev) => {
      const allSelected = group.cardIds.every((id) => prev.has(id));
      if (allSelected) return new Set();
      return new Set(group.cardIds);
    });
  }, []);

  // --- Straight flush finder (computed by backend) ---
  const sfBySuit = useMemo(() => {
    const handById = new Map(gameState.my_hand.map((c) => [c.id, c]));
    const result: Record<number, { label: string; cards: CardDTO[] }[]> = {};
    [0, 1, 2, 3].forEach((suit) => {
      result[suit] = (gameState.sf_options[suit] ?? []).map(({ label, cardIds }) => ({
        label,
        cards: cardIds.map((id) => handById.get(id)).filter(Boolean) as CardDTO[],
      }));
    });
    return result;
  }, [gameState.sf_options, gameState.my_hand]);

  const handleFlushSelect = useCallback((cards: CardDTO[]) => {
    setSelectedIds(new Set(cards.map((c) => c.id)));
  }, []);

  // --- Layout data ---
  const partner = gameState.players.find((p) => p.seat === 2);
  const leftOpp = gameState.players.find((p) => p.seat === 1);
  const rightOpp = gameState.players.find((p) => p.seat === 3);

  const ta = gameState.trick_actions;

  return (
    <div className="flex h-screen bg-background">
      {/* Main board area */}
      <div className="flex-1 flex flex-col p-6 gap-0">
        {/* Reconnection banner */}
        {connectionStatus === "reconnecting" && (
          <div className="flex items-center justify-center gap-2 py-2 bg-amber-50 border border-amber-200 rounded-lg mb-2">
            <div className="w-3 h-3 border-2 border-amber-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-xs font-medium text-amber-700">Reconnecting...</span>
          </div>
        )}
        {connectionStatus === "disconnected" && (
          <div className="flex items-center justify-center gap-2 py-2 bg-red-50 border border-red-200 rounded-lg mb-2">
            <span className="text-xs font-medium text-red-700">Connection lost. Please refresh the page.</span>
          </div>
        )}

        {/* Play area: all players + table in a centered grid with equal gaps */}
        <div className="flex-1 flex items-center justify-center">
          <div className="grid grid-cols-[auto_640px_auto] grid-rows-[auto_320px] gap-32 items-center justify-items-center">
            {/* Partner (top center, spans column 2) */}
            <div className="col-start-2 row-start-1">
              {partner && (
                <OpponentPanel player={partner} thinking={aiThinking === 2} />
              )}
            </div>

            {/* Left opponent */}
            <div className="col-start-1 row-start-2">
              {leftOpp && (
                <OpponentPanel player={leftOpp} thinking={aiThinking === 1} />
              )}
            </div>

            {/* Table surface — all trick actions inside */}
            <div className="col-start-2 row-start-2 relative w-[640px] h-[320px] border border-border rounded-2xl">
              {/* Partner (top edge) */}
              <div className="absolute top-3 left-0 right-0 flex justify-center">
                <TrickActionDisplay action={ta?.["2"] ?? null} />
              </div>
              {/* Left opp (left edge) */}
              <div className="absolute left-3 top-0 bottom-0 flex items-center">
                <TrickActionDisplay action={ta?.["1"] ?? null} />
              </div>
              {/* Right opp (right edge) */}
              <div className="absolute right-3 top-0 bottom-0 flex items-center">
                <TrickActionDisplay action={ta?.["3"] ?? null} />
              </div>
              {/* You (bottom edge) */}
              <div ref={tableSeat0Ref} className="absolute bottom-3 left-0 right-0 flex justify-center">
                {!flyingCards && <TrickActionDisplay action={ta?.["0"] ?? null} />}
              </div>
              {/* New trick label */}
              {gameState.is_leading && !ta?.["0"] && !ta?.["1"] && !ta?.["2"] && !ta?.["3"] && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-sm text-text-secondary">New trick</span>
                </div>
              )}
            </div>

            {/* Right opponent */}
            <div className="col-start-3 row-start-2">
              {rightOpp && (
                <OpponentPanel player={rightOpp} thinking={aiThinking === 3} />
              )}
            </div>
          </div>
        </div>

        {/* Your turn indicator + Controls */}
        <div className="py-2">
          <div className={`flex items-center justify-center gap-1.5 pb-2 ${!gameState.is_my_turn ? "invisible" : ""}`}>
            <div className="w-1.5 h-1.5 rounded-full bg-accent" />
            <span className="text-[13px] font-medium text-accent">
              {gameState.is_leading ? "Your turn to lead" : "Your turn to play"}
            </span>
          </div>
          <GameControls
            matchingCombo={matchingCombo}
            isLeading={gameState.is_leading}
            isMyTurn={gameState.is_my_turn}
            onPlay={handlePlay}
            onPass={handlePass}
          />
        </div>

        {/* Player hand with groups */}
        <div className="py-1">
          <PlayerHand
            cards={gameState.my_hand}
            selectedIds={selectedIds}
            onToggleCard={toggleCard}
            groups={localGroups}
            groupedCardIds={groupedCardIds}
            onGroupClick={handleGroupClick}
            hiddenIds={flyingCards ? new Set(flyingCards.cards.map((c) => c.id)) : undefined}
          />
        </div>

        {/* Hand toolbar */}
        <div className="py-2">
          <HandToolbar
            onFlushSelect={handleFlushSelect}
            sfBySuit={sfBySuit}
            onGroup={handleGroup}
            onUngroup={handleUngroup}
            canGroup={selectedIds.size > 0}
            canUngroup={localGroups.some((g) =>
              g.cardIds.some((cid) => selectedIds.has(cid))
            )}
          />
        </div>
      </div>

      {/* Sidebar: always visible, two sections */}
      <div className="w-[280px] bg-surface border-l border-border p-5 overflow-y-auto flex flex-col gap-6">
        {/* Groups section */}
        <div className="flex flex-col gap-2">
          <span className="text-[13px] font-semibold text-text-secondary tracking-wider uppercase">
            Groups
          </span>
          {localGroups.length === 0 ? (
            <span className="text-sm text-text-secondary">No groups yet</span>
          ) : (
            <div className="flex flex-col gap-1.5">
              {localGroups.map((group) => (
                <button
                  key={group.id}
                  className="flex items-center justify-between px-2.5 py-1.5 rounded-md bg-background border border-border hover:border-accent hover:text-accent text-left transition-colors"
                  onClick={() => handleGroupClick(group)}
                >
                  <span className="text-[13px] font-semibold text-foreground">
                    {group.comboName}
                  </span>
                  <span className="text-xs text-text-secondary">{group.cardIds.length}c</span>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="border-t border-border" />

        {/* Legal combos section — excludes cards already in groups */}
        <div className="flex flex-col gap-2">
          <span className="text-[13px] font-semibold text-text-secondary tracking-wider uppercase">
            Legal Combos
          </span>
          {gameState.is_my_turn ? (
            <ComboBrowser
              legalMoves={gameState.legal_moves}
              onSelectCombo={handleSelectCombo}
            />
          ) : (
            <span className="text-sm text-text-secondary">Not your turn</span>
          )}
        </div>
      </div>

      {/* Card fly animation overlay */}
      {flyingCards && (
        <FlyingCards
          cards={flyingCards.cards}
          fromRects={flyingCards.fromRects}
          toRect={flyingCards.toRect}
        />
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
