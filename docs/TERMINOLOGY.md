# Terminology

DART uses these terms consistently in docs and new code:

- `seat`: an absolute table position, numbered `0..3`. Seats `0` and `2` are
  partners; seats `1` and `3` are partners.
- `player`: the actor currently taking an action in the game engine API. In
  older engine code this often also means seat; new DART runtime code should
  prefer `seat` when referring to an absolute position.
- `team`: one partnership, either even seats (`0,2`) or odd seats (`1,3`).
- `latest learner team`: the team controlled by the current learner weights in
  mixed-opponent episodes.
- `trick role`: the actor's relative role in the current trick: leading, first
  responder, across, or last responder.

Compatibility note: some public APIs still expose `player` because the base
game environment and `Agent.act(env, player)` interface predate DART. Avoid
renaming those without a migration plan.
