"""Glicko-2 rating system implementation.

Reference: Mark Glickman, "Example of the Glicko-2 System" (2013).
http://www.glicko.net/glicko/glicko2.pdf

No external dependencies — pure math with Python stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Glicko-2 constants
TAU = 0.5  # system volatility constraint
EPSILON = 1e-6  # convergence threshold
DEFAULT_RATING = 1500.0
DEFAULT_RD = 350.0
DEFAULT_VOL = 0.06

# Glicko-2 scale factor (maps Glicko-1 scale to internal scale)
_Q = math.log(10) / 400  # ≈ 0.00575646


@dataclass
class GlickoPlayer:
    """A rated player in the Glicko-2 system."""
    rating: float = DEFAULT_RATING
    rd: float = DEFAULT_RD
    volatility: float = DEFAULT_VOL
    name: str = ""

    def to_dict(self) -> dict:
        d = {"rating": round(self.rating, 1), "rd": round(self.rd, 1),
             "volatility": round(self.volatility, 6)}
        if self.name:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, d: dict) -> GlickoPlayer:
        return cls(
            rating=d.get("rating", DEFAULT_RATING),
            rd=d.get("rd", DEFAULT_RD),
            volatility=d.get("volatility", DEFAULT_VOL),
            name=d.get("name", ""),
        )


def _to_glicko2(rating: float, rd: float) -> tuple[float, float]:
    """Convert Glicko-1 scale to Glicko-2 internal scale."""
    mu = (rating - DEFAULT_RATING) * _Q
    phi = rd * _Q
    return mu, phi


def _from_glicko2(mu: float, phi: float) -> tuple[float, float]:
    """Convert Glicko-2 internal scale back to Glicko-1."""
    rating = mu / _Q + DEFAULT_RATING
    rd = phi / _Q
    return rating, rd


def _g(phi: float) -> float:
    """Glicko-2 g function: reduces impact of opponents with high RD."""
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    """Expected score of player with rating mu against opponent mu_j."""
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def expected_score(player: GlickoPlayer, opponent: GlickoPlayer) -> float:
    """Win probability of player against opponent (Glicko-1 scale)."""
    mu, _ = _to_glicko2(player.rating, player.rd)
    mu_j, phi_j = _to_glicko2(opponent.rating, opponent.rd)
    return _E(mu, mu_j, phi_j)


def glicko2_update(
    player: GlickoPlayer,
    opponents: list[GlickoPlayer],
    outcomes: list[float],
) -> GlickoPlayer:
    """Update a player's rating after a rating period.

    Args:
        player: The player to update.
        opponents: List of opponents faced.
        outcomes: List of outcomes (1.0=win, 0.5=draw, 0.0=loss) per opponent.

    Returns:
        New GlickoPlayer with updated rating, RD, and volatility.
    """
    if not opponents:
        # No games: RD increases over time
        mu, phi = _to_glicko2(player.rating, player.rd)
        phi_new = math.sqrt(phi * phi + player.volatility * player.volatility)
        r, rd = _from_glicko2(mu, phi_new)
        return GlickoPlayer(rating=r, rd=min(rd, DEFAULT_RD),
                            volatility=player.volatility, name=player.name)

    mu, phi = _to_glicko2(player.rating, player.rd)

    # Step 3: Compute v (estimated variance)
    v_inv = 0.0
    delta_sum = 0.0
    for opp, s in zip(opponents, outcomes, strict=False):
        mu_j, phi_j = _to_glicko2(opp.rating, opp.rd)
        g_j = _g(phi_j)
        e_j = _E(mu, mu_j, phi_j)
        v_inv += g_j * g_j * e_j * (1.0 - e_j)
        delta_sum += g_j * (s - e_j)

    v = 1.0 / v_inv if v_inv > 0 else 1e10
    delta = v * delta_sum  # Step 4: improvement

    # Step 5: Compute new volatility (Illinois algorithm)
    sigma = player.volatility
    a = math.log(sigma * sigma)
    phi2 = phi * phi
    delta2 = delta * delta
    tau2 = TAU * TAU

    def f(x: float) -> float:
        ex = math.exp(x)
        d = phi2 + v + ex
        return (ex * (delta2 - phi2 - v - ex)) / (2.0 * d * d) - (x - a) / tau2

    # Find bounds
    A = a
    if delta2 > phi2 + v:
        B = math.log(delta2 - phi2 - v)
    else:
        k = 1
        while f(a - k * TAU) < 0:
            k += 1
        B = a - k * TAU

    # Illinois method
    fA = f(A)
    fB = f(B)
    for _ in range(100):
        if abs(B - A) < EPSILON:
            break
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB <= 0:
            A, fA = B, fB
        else:
            fA /= 2.0
        B, fB = C, fC

    sigma_new = math.exp(A / 2.0)

    # Step 6: Update RD
    phi_star = math.sqrt(phi2 + sigma_new * sigma_new)

    # Step 7: Update rating and RD
    phi_new = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    mu_new = mu + phi_new * phi_new * delta_sum

    r, rd = _from_glicko2(mu_new, phi_new)
    return GlickoPlayer(
        rating=r, rd=min(rd, DEFAULT_RD),
        volatility=sigma_new, name=player.name,
    )


def team_rating(p1: GlickoPlayer, p2: GlickoPlayer) -> GlickoPlayer:
    """Combine two teammates into a single virtual player for 2v2.

    Uses average rating, combined RD (lower = more certain), avg volatility.
    """
    avg_r = (p1.rating + p2.rating) / 2.0
    # Combined RD: both players contribute info, so uncertainty is reduced
    combined_rd = math.sqrt((p1.rd ** 2 + p2.rd ** 2) / 4.0)
    avg_vol = (p1.volatility + p2.volatility) / 2.0
    return GlickoPlayer(rating=avg_r, rd=combined_rd, volatility=avg_vol)
