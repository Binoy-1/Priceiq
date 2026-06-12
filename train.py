"""Train Q-Learning agent from the terminal — Phase 2.

Adds:
  --state-mode {basic,extended}        9-state vs 27-state (price × competitor)
  --auto-cost                          cost = 40% of mean price in source CSV
  --source-csv PATH                    used by --auto-cost
  --early-stop / --patience / --tol    plateau-based early stopping
  full convergence diagnostics in models/training_provenance.json
"""
from __future__ import annotations
import argparse
import json
import logging
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from agent import QLearningAgent, AgentConfig, SCHEMA_VERSION as AGENT_SCHEMA
from environment import DynamicPricingEnv, EnvConfig, SCHEMA_VERSION as ENV_SCHEMA

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("train")
ROOT = Path(__file__).resolve().parent


def detect_plateau(rewards: list[float], window: int, patience: int, tol: float) -> bool:
    """True if rolling-mean reward has changed by less than `tol` for `patience` windows."""
    if len(rewards) < window * (patience + 1):
        return False
    means = [float(np.mean(rewards[-window * (i + 1):-window * i or None]))
             for i in range(patience + 1)]
    diffs = [abs(means[i] - means[i + 1]) for i in range(patience)]
    return all(d < tol for d in diffs)


def run(cfg_a: AgentConfig, cfg_e: EnvConfig, episodes: int,
        early_stop: bool, patience: int, tol: float
        ) -> tuple[QLearningAgent, list[float], int, bool]:
    agent = QLearningAgent(cfg_a)
    env = DynamicPricingEnv(cfg_e)
    rewards: list[float] = []
    window = max(20, episodes // 50)
    stopped_early = False
    last_ep = episodes
    for ep in range(episodes):
        state = env.reset()
        total, done = 0.0, False
        while not done:
            action = agent.choose_action(int(state))
            next_state, reward, done, _ = env.step(action)
            agent.update(int(state), int(action), float(reward), int(next_state), bool(done))
            state = next_state
            total += float(reward)
        rewards.append(total)
        if (ep + 1) % max(1, episodes // 10) == 0:
            log.info("ep %4d/%d | ε=%.3f | mean(100)=%.3f",
                     ep + 1, episodes, agent.epsilon, float(np.mean(rewards[-100:])))
        if early_stop and detect_plateau(rewards, window, patience, tol):
            log.info("Plateau detected at ep %d (window=%d patience=%d tol=%.4f) — stopping early.",
                     ep + 1, window, patience, tol)
            stopped_early = True
            last_ep = ep + 1
            break
    return agent, rewards, last_ep, stopped_early


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--eps0", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--state-mode", choices=["basic", "extended"], default="basic")
    ap.add_argument("--coefficients", default=str(ROOT / "models" / "coefficients.npy"))
    ap.add_argument("--auto-cost", action="store_true",
                    help="Set env.cost = 0.4 * mean(price) from --source-csv")
    ap.add_argument("--source-csv", default=str(ROOT / "data" / "2025_retail_sample.csv"))
    ap.add_argument("--price-col", default="price")
    ap.add_argument("--cost", type=float, default=None,
                    help="Override env.cost explicitly (skips --auto-cost).")
    ap.add_argument("--early-stop", action="store_true")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.005)
    ap.add_argument("--out", default=str(ROOT / "models" / "q_table.npy"))
    ap.add_argument("--provenance", default=str(ROOT / "models" / "training_provenance.json"))
    args = ap.parse_args()

    # Coefficients
    coef_path = Path(args.coefficients)
    if coef_path.exists():
        a_coef, b_coef = np.load(coef_path).tolist()
        log.info("Loaded demand coefficients a=%.4f b=%.4f from %s", a_coef, b_coef, coef_path)
    else:
        a_coef, b_coef = 5.0, 1.5
        log.warning("No coefficients at %s — using defaults a=%.2f b=%.2f", coef_path, a_coef, b_coef)

    # Cost
    if args.cost is not None:
        cost = float(args.cost)
        cost_src = f"explicit --cost {cost:.3f}"
    elif args.auto_cost and Path(args.source_csv).exists():
        df = pd.read_csv(args.source_csv)
        pc = args.price_col if args.price_col in df.columns else next(
            (c for c in df.columns if "price" in c.lower()), None)
        if pc is None:
            raise SystemExit(f"--auto-cost: no price column in {args.source_csv}")
        cost = float(0.4 * df[pc].astype(float).mean())
        cost_src = f"auto: 0.4 × mean({pc}) from {args.source_csv} = {cost:.3f}"
    else:
        cost = 2.0
        cost_src = "default 2.0"
    log.info("Env cost = %.3f  (%s)", cost, cost_src)

    cfg_e = EnvConfig(a=float(a_coef), b=abs(float(b_coef)), cost=cost,
                      seed=args.seed, state_mode=args.state_mode)
    cfg_a = AgentConfig(alpha=args.alpha, gamma=args.gamma,
                        epsilon_start=args.eps0, seed=args.seed,
                        num_states=cfg_e.total_states)

    log.info("Training tabular Q-Learning | state_mode=%s | states=%d | actions=%d",
             cfg_e.state_mode, cfg_e.total_states, cfg_a.num_actions)

    t0 = time.time()
    agent, rewards, last_ep, stopped_early = run(
        cfg_a, cfg_e, args.episodes, args.early_stop, args.patience, args.tol
    )
    elapsed = time.time() - t0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, agent.q_table)
    log.info("Saved Q-table %s → %s", agent.q_table.shape, out)

    # Convergence diagnostics
    window = max(10, last_ep // 20)
    smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
    final_mean = float(np.mean(rewards[-100:])) if len(rewards) >= 100 else float(np.mean(rewards))
    plateau_delta = (float(np.mean(rewards[-100:]) - np.mean(rewards[-200:-100]))
                     if len(rewards) >= 200 else None)
    # Best moving-average peak
    peak_ep = int(np.argmax(smooth)) + window if len(smooth) > 0 else 0
    peak_val = float(np.max(smooth)) if len(smooth) > 0 else 0.0

    prov = {
        "experiment_id": agent.experiment_id,
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "agent_schema": AGENT_SCHEMA,
        "env_schema": ENV_SCHEMA,
        "episodes_requested": args.episodes,
        "episodes_run": last_ep,
        "stopped_early": stopped_early,
        "elapsed_seconds": round(elapsed, 2),
        "agent_config": cfg_a.__dict__,
        "env_config": {**cfg_e.__dict__, "total_states": cfg_e.total_states},
        "cost_source": cost_src,
        "coefficients_source": str(coef_path),
        "final_epsilon": agent.epsilon,
        "final_mean_reward_last_100": final_mean,
        "plateau_delta_last_vs_prev_100": plateau_delta,
        "peak_smoothed_reward": peak_val,
        "peak_episode": peak_ep,
        "smoothing_window": window,
        "reward_curve_raw": [float(r) for r in rewards],
        "reward_curve_smoothed": [float(x) for x in smooth],
        "q_table_shape": list(agent.q_table.shape),
        "out_q_table": str(out),
    }
    Path(args.provenance).parent.mkdir(parents=True, exist_ok=True)
    Path(args.provenance).write_text(json.dumps(prov, indent=2))
    log.info("Wrote provenance → %s", args.provenance)
    log.info("Done in %.1fs · ran %d episodes · final mean reward = %.3f",
             elapsed, last_ep, final_mean)


if __name__ == "__main__":
    main()
