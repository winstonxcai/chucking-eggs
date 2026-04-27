"""Partner-Visible PTIE experiment (pvguan).

Separate from the existing AZ pipeline. Two ablations:
  PV-AC:   critic sees own hand + partner hand + public info (no opponent hands)
  PV-PTIE: critic also sees opponent hands during training (privileged)

Entry points:
  encoders.py     — actor/critic state + action encoders
  actor_critic.py — ActorCriticNet
  rollout.py      — parallel per-player trajectory collector
  buffer.py       — per-player GAE + PPO buffer
  ppo.py          — PPO clipped loss + KL regulariser
  diagnostics.py  — per-iter telemetry
  agent.py        — PVGuanBot for eval/play
"""
