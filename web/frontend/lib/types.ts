export interface CardDTO {
  rank: number;
  suit: number;
  deck: number;
  id: string;
  display: string;
  rank_display: string;
  suit_symbol: string;
  is_wild?: boolean;
}

export interface ComboDTO {
  type: string;
  type_id: number;
  type_name: string;
  type_chinese: string;
  cards: CardDTO[];
  display?: string;
  key?: number;
  wild_count?: number;
  is_pass: boolean;
}

export interface PlayerDTO {
  seat: number;
  name: string;
  avatar: string | null;
  elo: number;
  card_count: number;
  is_out: boolean;
  is_teammate: boolean;
  is_human: boolean;
}

export interface TrickAction {
  type: "play" | "pass";
  combo?: ComboDTO;
}

export interface CardGroup {
  id: string;
  cardIds: string[];
  comboType: string;
  comboName: string;
}

export interface GameState {
  game_id: string;
  my_seat: number;
  current_player: number;
  is_my_turn: boolean;
  my_hand: CardDTO[];
  players: PlayerDTO[];
  trick_actions: Record<string, TrickAction | null>;
  is_leading: boolean;
  legal_moves: ComboDTO[];
  finish_order: number[];
  done: boolean;
  level_rank: number;
  rewards: Record<number, number> | null;
  groups: CardGroup[];
  sf_options: Record<string, { label: string; cardIds: string[] }[]>;
  turn_deadline_ms?: number;
  partner_hand?: CardDTO[];
  all_moves: ComboDTO[];
  trick_lead_seat: number | null;
  mode?: string;
}

export interface MovePlayedMsg {
  type: "move_played";
  seat: number;
  combo: ComboDTO;
  next_player: number;
  done: boolean;
}

export interface GameOverMsg {
  type: "game_over";
  finish_order: number[];
  rewards: Record<number, number>;
  players: { seat: number; name: string }[];
  elo_changes?: Record<string, { delta: number; before: number; after: number }>;
}

export interface AIThinkingMsg {
  type: "ai_thinking";
  seat: number;
}

export interface ErrorMsg {
  type: "error";
  message: string;
}

export interface RematchMsg {
  type: "rematch_created";
  game_id: string;
  room_code: string | null;
}

export type ServerMessage =
  | (GameState & { type: "game_state" })
  | MovePlayedMsg
  | GameOverMsg
  | AIThinkingMsg
  | ErrorMsg
  | RematchMsg;

// Phase 2: room / lobby types

export type RoomMode = "solo" | "duo" | "quad";

export interface RoomSeatInfo {
  seat: number;
  is_human: boolean;
  connected: boolean;
  name: string;
}

export interface RoomStatus {
  game_id: string;
  mode: RoomMode;
  room_code: string | null;
  started: boolean;
  seats: RoomSeatInfo[];
}

export interface CreateRoomResponse {
  game_id: string;
  room_code: string | null;
  seat: number;
  reconnect_token: string;
}

export interface JoinRoomResponse {
  game_id: string;
  room_code: string;
  seat: number;
  reconnect_token: string;
}
