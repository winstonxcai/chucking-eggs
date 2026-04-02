"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Crown, Hand, HelpCircle, LayoutList } from "lucide-react";
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
  onRematch?: () => void;
  isMultiplayer?: boolean;
  onCreateGroup: (cardIds: string[], comboType: string, comboName: string) => void;
  onDeleteGroup: (groupId: string) => void;
  latestError?: { message: string; key: number } | null;
}

/** Render a single trick action (cards or "Pass") */
function TrickActionDisplay({ action }: { action: TrickAction | null }) {
  if (!action) return null;
  if (action.type === "pass") {
    return (
      <span className="flex items-center gap-1 px-2 py-0.5 rounded-full border border-border text-xs text-text-secondary">
        <Hand className="w-3 h-3 shrink-0" />
        Pass
      </span>
    );
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
  onRematch,
  isMultiplayer,
  onCreateGroup,
  onDeleteGroup,
  latestError,
}: GameBoardProps) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [flyingCards, setFlyingCards] = useState<{
    cards: CardDTO[];
    fromRects: DOMRect[];
    toRect: DOMRect;
  } | null>(null);
  const [toasts, setToasts] = useState<{ id: number; text: string }[]>([]);
  const [gameOverDismissed, setGameOverDismissed] = useState(false);
  const tableSeat0Ref = useRef<HTMLDivElement>(null);
  const flyingStartRef = useRef<number | null>(null);

  // Sidebar accordion state
  const [groupsOpen, setGroupsOpen] = useState(true);
  const [legalOpen, setLegalOpen] = useState(true);
  const [allOpen, setAllOpen] = useState(false);

  // Mobile state
  const [compactHand, setCompactHand] = useState(false);
  const [mobileComboOpen, setMobileComboOpen] = useState(false);

  // Reset dismissed state when a new game starts
  useEffect(() => {
    if (!gameOver) setGameOverDismissed(false);
  }, [gameOver]);

  useEffect(() => {
    const mq = window.matchMedia("(max-width: 1023px)");
    const update = (e: MediaQueryListEvent | MediaQueryList) => setCompactHand(e.matches);
    update(mq);
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);

  // Optimistic local groups — updated immediately on Group/Ungroup, synced from server on game_state
  const [localGroups, setLocalGroups] = useState<CardGroup[]>(gameState.groups);
  useEffect(() => {
    setLocalGroups((prev) => {
      const serverCardSets = new Set(
        gameState.groups.map((g) => [...g.cardIds].sort().join(","))
      );
      const pendingTemps = prev.filter(
        (g) =>
          g.id.startsWith("grp-temp-") &&
          !serverCardSets.has([...g.cardIds].sort().join(","))
      );
      return [...gameState.groups, ...pendingTemps];
    });
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
  }, [gameState, flyingCards]); // eslint-disable-line react-hooks/exhaustive-deps

  // On server error: immediately clear any stuck flying animation and show a toast
  useEffect(() => {
    if (!latestError) return;
    const { key, message } = latestError;
    setFlyingCards(null);
    flyingStartRef.current = null;
    setToasts((prev) => [...prev, { id: key, text: message }]);
    const t = setTimeout(() => {
      setToasts((prev) => prev.filter((toast) => toast.id !== key));
    }, 3000);
    return () => clearTimeout(t);
  }, [latestError?.key]); // eslint-disable-line react-hooks/exhaustive-deps

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

  const handleSaveState = useCallback(() => {
    const blob = new Blob([JSON.stringify(gameState, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `game-state-${gameState.game_id}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [gameState]);

  // --- Layout data ---
  const partner = gameState.players.find((p) => p.seat === 2);
  const leftOpp = gameState.players.find((p) => p.seat === 1);
  const rightOpp = gameState.players.find((p) => p.seat === 3);

  const ta = gameState.trick_actions;

  return (
    <div className="flex h-[100dvh] bg-background">
      {/* Main board area */}
      <div className="flex-1 flex flex-col p-2 lg:p-6 gap-0 relative">
        {/* Help button */}
        <a
          href="https://www.pagat.com/climbing/guandan.html"
          target="_blank"
          rel="noopener noreferrer"
          className="absolute top-2 right-2 lg:top-4 lg:right-4 z-20 text-text-secondary hover:text-foreground transition-colors"
          title="Game rules"
        >
          <HelpCircle size={18} />
        </a>
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

        {/* Play area: all players + table in a centered grid */}
        <div className="flex-1 flex items-center justify-center">
          <div className="grid grid-cols-[auto_minmax(0,640px)_auto] gap-1 lg:gap-32 items-center justify-items-center">
            {/* Partner (top center, spans column 2) */}
            <div className="col-start-2 row-start-1">
              {partner && (
                <div className={gameState.current_player === 2 ? "ring-2 ring-accent rounded-xl" : ""}>
                  <OpponentPanel player={partner} thinking={aiThinking === 2} revealedHand={gameState.partner_hand} />
                </div>
              )}
            </div>

            {/* Left opponent */}
            <div className="col-start-1 row-start-2">
              {leftOpp && (
                <div className={gameState.current_player === 1 ? "ring-2 ring-accent rounded-xl" : ""}>
                  <OpponentPanel player={leftOpp} thinking={aiThinking === 1} />
                </div>
              )}
            </div>

            {/* Table surface — all trick actions inside */}
            <div className="col-start-2 row-start-2 relative w-full h-[150px] lg:w-[640px] lg:h-[320px] border border-border rounded-2xl">
              {/* Partner (top edge) */}
              <div data-testid="trick-seat-2" className="absolute top-3 left-0 right-0 flex justify-center">
                <div className="relative inline-flex">
                  <TrickActionDisplay action={ta?.["2"] ?? null} />
                  {gameState.trick_lead_seat === 2 && <Crown className="absolute -top-2 -right-2 w-3.5 h-3.5 text-yellow-400" />}
                </div>
              </div>
              {/* Left opp (left edge) */}
              <div data-testid="trick-seat-1" className="absolute left-3 top-0 bottom-0 flex items-center">
                <div className="relative inline-flex">
                  <TrickActionDisplay action={ta?.["1"] ?? null} />
                  {gameState.trick_lead_seat === 1 && <Crown className="absolute -top-2 -right-2 w-3.5 h-3.5 text-yellow-400" />}
                </div>
              </div>
              {/* Right opp (right edge) */}
              <div data-testid="trick-seat-3" className="absolute right-3 top-0 bottom-0 flex items-center">
                <div className="relative inline-flex">
                  <TrickActionDisplay action={ta?.["3"] ?? null} />
                  {gameState.trick_lead_seat === 3 && <Crown className="absolute -top-2 -right-2 w-3.5 h-3.5 text-yellow-400" />}
                </div>
              </div>
              {/* You (bottom edge) */}
              <div ref={tableSeat0Ref} data-testid="trick-seat-0" className="absolute bottom-3 left-0 right-0 flex justify-center">
                <div className="relative inline-flex">
                  {!flyingCards && <TrickActionDisplay action={ta?.["0"] ?? null} />}
                  {gameState.trick_lead_seat === 0 && <Crown className="absolute -top-2 -right-2 w-3.5 h-3.5 text-yellow-400" />}
                </div>
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
                <div className={gameState.current_player === 3 ? "ring-2 ring-accent rounded-xl" : ""}>
                  <OpponentPanel player={rightOpp} thinking={aiThinking === 3} />
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Your turn indicator + Controls */}
        <div className="py-1 lg:py-2">
          <div className={`flex items-center justify-center gap-1.5 pb-1 lg:pb-2 ${!gameState.is_my_turn ? "invisible" : ""}`}>
            <div className="w-1.5 h-1.5 rounded-full bg-accent" />
            <span className="text-[13px] font-medium text-accent">
              {gameState.is_leading ? "Your turn to lead" : "Your turn to play"}
            </span>
          </div>
          <GameControls
            matchingCombo={matchingCombo}
            isLeading={gameState.is_leading}
            isMyTurn={gameState.is_my_turn}
            hasSelection={selectedIds.size > 0}
            turnDeadlineMs={gameState.turn_deadline_ms}
            onPlay={handlePlay}
            onPass={handlePass}
            onUnselect={() => setSelectedIds(new Set())}
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
            compact={compactHand}
          />
        </div>

        {/* Hand toolbar + mobile combos button */}
        <div className="py-0.5 lg:py-2 flex items-center justify-center gap-2">
          <HandToolbar
            onFlushSelect={handleFlushSelect}
            sfBySuit={sfBySuit}
            onGroup={handleGroup}
            onUngroup={handleUngroup}
            canGroup={selectedIds.size > 0 && !Array.from(selectedIds).some((id) => groupedCardIds.has(id))}
            canUngroup={localGroups.some((g) =>
              g.cardIds.some((cid) => selectedIds.has(cid))
            )}
          />
          <button
            className="lg:hidden flex items-center gap-1 px-2.5 py-1 bg-surface border border-border rounded-md text-xs font-medium text-foreground"
            onClick={() => setMobileComboOpen(true)}
          >
            <LayoutList size={13} />
            Combos
          </button>
        </div>
      </div>

      {/* Sidebar: always visible on desktop, hidden on mobile */}
      <div className="w-[280px] bg-surface border-l border-border p-5 overflow-y-auto hidden lg:flex flex-col gap-4">
        {/* Groups section */}
        <div className="flex flex-col gap-2">
          <button
            onClick={() => setGroupsOpen((o) => !o)}
            className="flex items-center justify-between w-full text-[13px] font-semibold text-text-secondary tracking-wider uppercase"
          >
            <span>Groups</span>
            <span className="text-xs">{groupsOpen ? "▾" : "▸"}</span>
          </button>
          {groupsOpen && (
            localGroups.length === 0 ? (
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
            )
          )}
        </div>

        <div className="border-t border-border" />

        {/* Legal combos section — excludes cards already in groups */}
        <div className="flex flex-col gap-2">
          <button
            onClick={() => setLegalOpen((o) => !o)}
            className="flex items-center justify-between w-full text-[13px] font-semibold text-text-secondary tracking-wider uppercase"
          >
            <span>Legal Combos</span>
            <span className="text-xs">{legalOpen ? "▾" : "▸"}</span>
          </button>
          {legalOpen && (
            gameState.is_my_turn ? (
              <ComboBrowser
                legalMoves={gameState.legal_moves.filter(
                  (m) => !m.cards.every((c) => groupedCardIds.has(c.id))
                )}
                onSelectCombo={handleSelectCombo}
              />
            ) : (
              <span className="text-sm text-text-secondary">Not your turn</span>
            )
          )}
        </div>

        <div className="border-t border-border" />

        {/* All combos section — all valid combos from hand, ignoring current trick */}
        <div className="flex flex-col gap-2">
          <button
            onClick={() => setAllOpen((o) => !o)}
            className="flex items-center justify-between w-full text-[13px] font-semibold text-text-secondary tracking-wider uppercase"
          >
            <span>All Combos</span>
            <span className="text-xs">{allOpen ? "▾" : "▸"}</span>
          </button>
          {allOpen && (
            <ComboBrowser
              legalMoves={(gameState.all_moves ?? []).filter(
                (m) => !m.cards.every((c) => groupedCardIds.has(c.id))
              )}
              onSelectCombo={handleSelectCombo}
            />
          )}
        </div>

      </div>

      {/* Mobile combo bottom sheet */}
      {mobileComboOpen && (
        <div className="lg:hidden fixed inset-0 z-50 flex flex-col justify-end">
          <div
            className="absolute inset-0 bg-black/40"
            onClick={() => setMobileComboOpen(false)}
          />
          <div className="relative bg-surface rounded-t-2xl p-5 max-h-[60dvh] overflow-y-auto flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-foreground">Legal Combos</span>
              <button
                onClick={() => setMobileComboOpen(false)}
                className="text-xs text-text-secondary hover:text-foreground transition-colors"
              >
                Close
              </button>
            </div>
            {gameState.is_my_turn ? (
              <ComboBrowser
                legalMoves={gameState.legal_moves.filter(
                  (m) => !m.cards.every((c) => groupedCardIds.has(c.id))
                )}
                onSelectCombo={(combo) => {
                  handleSelectCombo(combo);
                  setMobileComboOpen(false);
                }}
              />
            ) : (
              <span className="text-sm text-text-secondary">Not your turn</span>
            )}
          </div>
        </div>
      )}

      {/* Toast notifications — server error feedback */}
      {toasts.length > 0 && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-[60] flex flex-col items-center gap-2 pointer-events-none">
          {toasts.map((toast) => (
            <div
              key={toast.id}
              className="bg-foreground text-background text-sm font-medium px-4 py-2.5 rounded-xl shadow-lg"
            >
              {toast.text}
            </div>
          ))}
        </div>
      )}

      {/* Card fly animation overlay */}
      {flyingCards && (
        <FlyingCards
          cards={flyingCards.cards}
          fromRects={flyingCards.fromRects}
          toRect={flyingCards.toRect}
        />
      )}

      {/* Game over modal */}
      {gameOver && !gameOverDismissed && (
        <GameOverModal
          data={gameOver}
          humanSeat={gameState.my_seat}
          onPlayAgain={isMultiplayer && onRematch ? onRematch : onPlayAgain}
          isMultiplayer={isMultiplayer}
          onSaveState={handleSaveState}
          onDismiss={() => setGameOverDismissed(true)}
        />
      )}
    </div>
  );
}
