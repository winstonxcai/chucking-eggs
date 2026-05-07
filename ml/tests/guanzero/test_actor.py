from __future__ import annotations

from guandan.guanzero.actor import play_episode
from guandan.guanzero.config import QNetConfig
from guandan.guanzero.encoder import StateActionEncoder
from guandan.guanzero.q_network import init_seat_nets
from guandan.guanzero.returns import TrainSample


def test_one_episode_returns_team_signed_samples():
    encoder = StateActionEncoder()
    q_nets = init_seat_nets(QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2))
    samples = play_episode(
        q_nets=q_nets,
        encoder=encoder,
        epsilon=1.0,  # pure random play for fastest termination
        seed=7,
        device="cpu",
        gamma=1.0,
    )
    assert len(samples) > 0
    assert all(isinstance(s, TrainSample) for s in samples)
    # Team-signed: per-player returns should be ±1, summing to zero across teams
    by_player = {p: [] for p in range(4)}
    for s in samples:
        by_player[s.player].append(s.mc_return)
    for p, vals in by_player.items():
        if vals:
            assert all(v == vals[0] for v in vals)  # constant across player's traj (gamma=1)
            assert vals[0] in (1.0, -1.0, 2/3, -2/3, 1/3, -1/3)
    # Teammates share sign
    if by_player[0] and by_player[2]:
        assert (by_player[0][0] > 0) == (by_player[2][0] > 0)
    if by_player[1] and by_player[3]:
        assert (by_player[1][0] > 0) == (by_player[3][0] > 0)
