export const STORAGE_KEYS = {
  // sessionStorage — active game state
  GAME_ID: "gd_game_id",
  RECONNECT_TOKEN: "gd_reconnect_token",
  SEAT: "gd_seat",
  // localStorage — player identity
  PLAYER_ID: "ce_player_id",
  PLAYER_TOKEN: "ce_player_token",
  USERNAME: "ce_username",
  ELO: "ce_elo",
} as const;
