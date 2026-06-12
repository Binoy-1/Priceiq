"""Dynamic pricing Gym environment — research-grade hardened.

Supports two state encodings:
  - basic     : state = price_bin                          (num_states = num_price_bins)
  - extended  : state = price_bin * num_competitor_bins
                        + competitor_relation_bin
                (num_states = num_price_bins * num_competitor_bins)

Set cfg.state_mode = "extended" to enable the richer (price × competitor)
encoding. The default ("basic") preserves the original 9-state behavior so
previously trained Q-tables remain compatible.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import gym
from gym import spaces

SCHEMA_VERSION = "2.0.0"

logger = logging.getLogger(__name__)


# ── Exception taxonomy ────────────────────────────────────────────────────────
class EnvironmentError(Exception):
    """Base for environment-domain errors."""

class InvalidCoefficientsError(EnvironmentError):
    """Demand coefficients produce degenerate or negative demand everywhere."""

class PriceConstraintError(EnvironmentError):
    """Price floor >= price ceiling — environment is misconfigured."""


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class EnvConfig:
    """All environment hyper-parameters in one validated dataclass."""
    a:                   float = 5.0
    b:                   float = 1.5
    price_floor:         float = 1.0
    price_ceiling:       float = 20.0
    cost:                float = 2.0
    step_size:           float = 1.0
    max_steps:           int   = 60
    noise_std:           float = 1.0
    num_states:          int   = 9      # legacy alias for num_price_bins
    seed:                int   = 42

    # ── Extended-state options ───────────────────────────────────────────
    state_mode:          str   = "basic"        # "basic" or "extended"
    num_competitor_bins: int   = 3              # we_cheaper / parity / we_pricier
    competitor_mean:     float = 10.0           # anchor for competitor random walk
    competitor_std:      float = 1.5            # per-step noise
    season_amplitude:    float = 0.08           # ±8% annual seasonal swing
    season_periodicity:  int   = 52             # one cycle per N steps

    def __post_init__(self) -> None:
        if self.price_floor >= self.price_ceiling:
            raise PriceConstraintError(
                f"price_floor={self.price_floor} >= price_ceiling={self.price_ceiling}"
            )
        if self.cost < 0:
            raise PriceConstraintError(f"cost={self.cost} must be non-negative")
        if self.cost >= self.price_floor:
            logger.warning(
                "cost=%.2f >= price_floor=%.2f: agent cannot make profit at floor",
                self.cost, self.price_floor,
            )
        if self.state_mode not in ("basic", "extended"):
            raise PriceConstraintError(f"state_mode={self.state_mode!r} not in (basic, extended)")
        mid = (self.price_floor + self.price_ceiling) / 2.0
        mid_demand = np.exp(self.a - self.b * np.log(mid))
        if mid_demand <= 0:
            raise InvalidCoefficientsError(
                f"Demand at midpoint price {mid:.2f} is non-positive: {mid_demand:.4f}. "
                "Check fitted coefficients."
            )

    # ── Derived ──────────────────────────────────────────────────────────
    @property
    def num_price_bins(self) -> int:
        return int(self.num_states)

    @property
    def total_states(self) -> int:
        if self.state_mode == "extended":
            return int(self.num_price_bins * self.num_competitor_bins)
        return int(self.num_price_bins)


# ── Environment ───────────────────────────────────────────────────────────────
class DynamicPricingEnv(gym.Env):
    """Log-log demand pricing environment with optional extended state."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, cfg: EnvConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(self.cfg.total_states)

        self.price_levels: np.ndarray = np.linspace(
            self.cfg.price_floor, self.cfg.price_ceiling, self.cfg.num_price_bins
        )

        # episode state — initialised by reset()
        self.current_price_idx: int   = 0
        self.current_price:     float = self.price_levels[0]
        self.competitor_price:  float = self.cfg.competitor_mean
        self.steps:             int   = 0

        logger.info(
            "DynamicPricingEnv ready | mode=%s | states=%d | prices=%s | a=%.4f | b=%.4f",
            self.cfg.state_mode, self.cfg.total_states,
            np.round(self.price_levels, 2).tolist(), self.cfg.a, self.cfg.b,
        )

    # ── State encoding ────────────────────────────────────────────────────────
    def _competitor_bin(self) -> int:
        """0 = we are cheaper, 1 = parity, 2 = we are pricier."""
        gap = self.current_price - self.competitor_price
        # parity band = 5% of price range
        band = 0.05 * (self.cfg.price_ceiling - self.cfg.price_floor)
        if gap < -band:
            return 0
        if gap > band:
            return 2
        return 1

    def _encode_state(self) -> int:
        if self.cfg.state_mode == "extended":
            return int(self.current_price_idx * self.cfg.num_competitor_bins + self._competitor_bin())
        return int(self.current_price_idx)

    def _season_factor(self) -> float:
        return float(
            1.0 + self.cfg.season_amplitude * np.sin(
                2 * np.pi * self.steps / max(1, self.cfg.season_periodicity)
            )
        )

    # ── Gym interface ─────────────────────────────────────────────────────────
    def reset(self) -> int:
        self.current_price_idx = int(self._rng.integers(0, self.cfg.num_price_bins))
        self.current_price     = float(self.price_levels[self.current_price_idx])
        self.competitor_price  = float(
            np.clip(self._rng.normal(self.cfg.competitor_mean, self.cfg.competitor_std),
                    self.cfg.price_floor, self.cfg.price_ceiling)
        )
        self.steps = 0
        return self._encode_state()

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        # Update price
        if action == 0:
            self.current_price = max(self.cfg.price_floor, self.current_price - self.cfg.step_size)
        elif action == 2:
            self.current_price = min(self.cfg.price_ceiling, self.current_price + self.cfg.step_size)

        self.current_price_idx = int(
            np.argmin(np.abs(self.price_levels - self.current_price))
        )

        # Competitor random walk (anchored mean-reversion)
        drift = 0.2 * (self.cfg.competitor_mean - self.competitor_price)
        self.competitor_price = float(np.clip(
            self.competitor_price + drift + self._rng.normal(0.0, self.cfg.competitor_std * 0.3),
            self.cfg.price_floor, self.cfg.price_ceiling,
        ))

        # Demand: log-log + season factor + competitor share factor in extended mode
        noise = float(self._rng.normal(0.0, self.cfg.noise_std))
        season = self._season_factor()
        base_demand = max(0.0, np.exp(self.cfg.a - self.cfg.b * np.log(self.current_price)) + noise)
        if self.cfg.state_mode == "extended":
            # cheaper than competitor → win share; pricier → lose share
            gap = self.competitor_price - self.current_price
            share = 1.0 / (1.0 + np.exp(-gap))   # logistic
            share_mult = 0.6 + 0.8 * share        # in [0.6, 1.4]
            demand = base_demand * season * share_mult
        else:
            demand = base_demand * season

        profit_margin = max(0.0, self.current_price - self.cfg.cost)
        reward = (profit_margin * demand) / 1_000.0

        self.steps += 1
        done = self.steps >= self.cfg.max_steps

        info = {
            "price":            self.current_price,
            "competitor_price": self.competitor_price,
            "season_factor":    season,
            "demand":           demand,
            "step":             self.steps,
        }
        return self._encode_state(), reward, done, info

    def render(self, mode: str = "human") -> None:
        logger.debug("price=%.2f | comp=%.2f | steps=%d",
                     self.current_price, self.competitor_price, self.steps)

    # ── Analysis helpers ──────────────────────────────────────────────────────
    def expected_revenue(self, price: float, n_samples: int = 200) -> float:
        rng = np.random.default_rng(self.cfg.seed)
        noises = rng.normal(0.0, self.cfg.noise_std, size=n_samples)
        demands = np.maximum(
            0.0, np.exp(self.cfg.a - self.cfg.b * np.log(price)) + noises
        )
        return float(np.mean(price * demands))
