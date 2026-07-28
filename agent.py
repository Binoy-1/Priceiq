"""Q-Learning agent for dynamic pricing — research-grade hardened."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

import numpy as np

SCHEMA_VERSION = "1.0.0"

logger = logging.getLogger(__name__)


# ── Exception taxonomy ────────────────────────────────────────────────────────
class AgentError(Exception):
    """Base for all agent-domain errors."""

class InvalidStateError(AgentError):
    """State index is outside the valid range."""

class InvalidActionError(AgentError):
    """Action index is outside the valid range."""


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class AgentConfig:
    """Hyperparameters for QLearningAgent. [REPRODUCIBILITY]"""
    num_states:     int   = 9
    num_actions:    int   = 3
    alpha:          float = 0.1      # learning rate
    gamma:          float = 0.99     # discount factor
    epsilon_start:  float = 0.3      # initial exploration rate
    epsilon_end:    float = 0.01     # minimum exploration rate
    epsilon_decay:  float = 0.995    # multiplicative decay per step
    seed:           int   = 42


# ── Agent ─────────────────────────────────────────────────────────────────────
class QLearningAgent:
    """
    Tabular Q-Learning agent with local RNG and strict state validation.

    Changes vs original
    -------------------
    [REPRODUCIBILITY] Local RNG (np.random.default_rng) — never mutates global state.
    [REPRODUCIBILITY] experiment_id = full UUID, schema_version stored.
    [CORRECTNESS]     State / action bounds validated before every Q-table access.
    [ROBUSTNESS]      Custom exception taxonomy replaces bare IndexError propagation.
    [ROBUSTNESS]      All print() replaced by logging.
    """

    def __init__(self, cfg: AgentConfig | None = None) -> None:
        self.cfg = cfg or AgentConfig()
        # [REPRODUCIBILITY] local RNG — does not mutate np.random global state
        self._rng = np.random.default_rng(self.cfg.seed)
        self.q_table: np.ndarray = np.zeros(
            (self.cfg.num_states, self.cfg.num_actions), dtype=np.float64
        )
        self.epsilon: float = self.cfg.epsilon_start
        self.experiment_id: str = str(uuid.uuid4())   # [REPRODUCIBILITY] full UUID
        self.schema_version: str = SCHEMA_VERSION
        logger.info(
            "QLearningAgent initialised | experiment_id=%s | states=%d | actions=%d",
            self.experiment_id, self.cfg.num_states, self.cfg.num_actions,
        )

    # ── helpers ───────────────────────────────────────────────────────────────
    def _validate_state(self, state: int) -> None:
        """[ROBUSTNESS] Raise InvalidStateError instead of bare IndexError."""
        if not (0 <= state < self.cfg.num_states):
            raise InvalidStateError(
                f"state={state} outside [0, {self.cfg.num_states - 1}]"
            )

    def _validate_action(self, action: int) -> None:
        """[ROBUSTNESS] Raise InvalidActionError instead of bare IndexError."""
        if not (0 <= action < self.cfg.num_actions):
            raise InvalidActionError(
                f"action={action} outside [0, {self.cfg.num_actions - 1}]"
            )

    # ── public API ────────────────────────────────────────────────────────────
    def choose_action(self, state: int) -> int:
        """
        Epsilon-greedy action selection using local RNG.

        [REPRODUCIBILITY] Uses self._rng — no global np.random calls.
        [ROBUSTNESS]      Validates state before Q-table lookup.
        """
        self._validate_state(state)
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.cfg.num_actions))
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """
        Bellman update: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s′,·) − Q(s,a)].

        [CORRECTNESS]    Uses max over next_state Q-values (not argmax index).
        [ROBUSTNESS]     Validates all indices before access.
        [REPRODUCIBILITY] Epsilon decay is deterministic — no RNG involved.
        """
        self._validate_state(state)
        self._validate_action(action)
        self._validate_state(next_state)

        future_value = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_target = reward + self.cfg.gamma * future_value
        td_error  = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.cfg.alpha * td_error

        # Epsilon decay — deterministic, no RNG
        if self.epsilon > self.cfg.epsilon_end:
            self.epsilon = max(self.cfg.epsilon_end,
                               self.epsilon * self.cfg.epsilon_decay)

    def get_policy(self) -> np.ndarray:
        """Return greedy action for every state. Shape: (num_states,)."""
        return np.argmax(self.q_table, axis=1)

    def best_price_state(self) -> int:
        """Return the state index with the highest max Q-value."""
        return int(np.argmax(np.max(self.q_table, axis=1)))
