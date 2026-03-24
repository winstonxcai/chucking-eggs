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

export interface TrickPlay {
  seat: number;
  player_name: string;
  combo: ComboDTO;
  is_pass: boolean;
}

export interface CurrentTrick {
  plays: TrickPlay[];
  leader: number;
}

export interface GameState {
  game_id: string;
  my_seat: number;
  current_player: number;
  is_my_turn: boolean;
  my_hand: CardDTO[];
  players: PlayerDTO[];
  current_trick: CurrentTrick | null;
  is_leading: boolean;
  legal_moves: ComboDTO[];
  finish_order: number[];
  done: boolean;
  level_rank: number;
  rewards: Record<number, number> | null;
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
}

export interface AIThinkingMsg {
  type: "ai_thinking";
  seat: number;
}

export interface ErrorMsg {
  type: "error";
  message: string;
}

export type ServerMessage =
  | (GameState & { type: "game_state" })
  | MovePlayedMsg
  | GameOverMsg
  | AIThinkingMsg
  | ErrorMsg;
