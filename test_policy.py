"""Evaluate a trained Q-table — Phase 2.

Adds:
  - matches Q-table shape to env automatically (basic vs extended)
  - --holdout-csv: simulate the learned greedy policy on observed
    history (chronological 80/20 split by default), comparing against
    the actual realised revenue + fixed-price baselines.
  - lift % vs each baseline written to evaluation_report.json
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from environment import DynamicPricingEnv, EnvConfig
from utils.helpers import bin_price_to_state, predict_demand, ACTION_DELTA

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("test_policy")
ROOT = Path(__file__).resolve().parent


def run_policy(env: DynamicPricingEnv, choose_action, episodes: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rewards, prices, demands, zeros = [], [], [], 0
    for _ in range(episodes):
        state = env.reset()
        total, done = 0.0, False
        while not done:
            action = int(choose_action(int(state), rng))
            next_state, reward, done, info = env.step(action)
            total += float(reward)
            prices.append(info["price"]); demands.append(info["demand"])
            if reward == 0.0:
                zeros += 1
            state = next_state
        rewards.append(total)
    return {
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward":  float(np.std(rewards)),
        "mean_price":  float(np.mean(prices)),
        "mean_demand": float(np.mean(demands)),
        "zero_reward_count": int(zeros),
    }


def holdout_evaluation(q_table: np.ndarray, holdout: pd.DataFrame, cfg: EnvConfig,
                       price_col: str, qty_col: str | None,
                       revenue_col: str | None) -> dict:
    """Walk the held-out chronological slice with the learned greedy policy
    and compare against the actual realised revenue at the observed price.

    Demand for the policy's chosen price is predicted from the fitted log-log
    model (cfg.a, cfg.b) — this avoids cheating by reusing the observed price's
    realised demand.
    """
    if qty_col is None and revenue_col is not None:
        holdout = holdout.copy()
        holdout["__qty__"] = holdout[revenue_col].astype(float) / holdout[price_col].astype(float).clip(1e-9)
        qty_col = "__qty__"
    if qty_col is None:
        raise ValueError("holdout: need a quantity or revenue column")

    prices_obs = holdout[price_col].astype(float).values
    qty_obs    = holdout[qty_col].astype(float).values
    rev_obs    = prices_obs * qty_obs

    # Walk policy starting from first observed price
    price = float(prices_obs[0])
    pol_prices, pol_demands, pol_rev = [], [], []
    for _ in range(len(holdout)):
        state = bin_price_to_state(price, cfg.price_floor, cfg.price_ceiling, cfg.num_price_bins)
        # Extended-state Q-tables not directly applicable on raw history (no competitor
        # observations) — fall back to the price-bin row by averaging across competitor bins.
        if q_table.shape[0] == cfg.num_price_bins * cfg.num_competitor_bins:
            base = state * cfg.num_competitor_bins
            row = q_table[base:base + cfg.num_competitor_bins].mean(axis=0)
        else:
            row = q_table[state]
        action = int(np.argmax(row))
        delta = ACTION_DELTA.get(action, 0.0)
        price = float(np.clip(price + delta, cfg.price_floor, cfg.price_ceiling))
        d = predict_demand(price, cfg.a, cfg.b)
        pol_prices.append(price); pol_demands.append(d); pol_rev.append(price * d)

    def stats(arr): return {"mean": float(np.mean(arr)), "sum": float(np.sum(arr))}

    learned_rev = float(np.sum(pol_rev))
    actual_rev  = float(np.sum(rev_obs))

    # Fixed-price baselines on the same horizon
    def baseline_at(p: float) -> float:
        return float(np.sum(p * predict_demand(p, cfg.a, cfg.b) for p in [p] * len(holdout)))

    baselines = {
        "actual_observed":  actual_rev,
        "fixed_floor":      float(np.sum([cfg.price_floor *
                                          predict_demand(cfg.price_floor, cfg.a, cfg.b)] * len(holdout))),
        "fixed_midpoint":   float(np.sum([((cfg.price_floor + cfg.price_ceiling) / 2) *
                                          predict_demand((cfg.price_floor + cfg.price_ceiling) / 2,
                                                         cfg.a, cfg.b)] * len(holdout))),
        "fixed_ceiling":    float(np.sum([cfg.price_ceiling *
                                          predict_demand(cfg.price_ceiling, cfg.a, cfg.b)] * len(holdout))),
    }
    lift = {k: (float((learned_rev - v) / v * 100) if abs(v) > 1e-9 else float("inf"))
            for k, v in baselines.items()}

    return {
        "horizon_periods":     int(len(holdout)),
        "learned_total_rev":   learned_rev,
        "learned_price":       stats(pol_prices),
        "learned_demand":      stats(pol_demands),
        "baselines_total_rev": baselines,
        "lift_pct_vs_baseline": lift,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--q", default=str(ROOT / "models" / "q_table.npy"))
    ap.add_argument("--coefficients", default=str(ROOT / "models" / "coefficients.npy"))
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--state-mode", choices=["basic", "extended", "auto"], default="auto")
    ap.add_argument("--holdout-csv", default=None,
                    help="Optional: evaluate on chronological tail of this CSV.")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--price-col", default="price")
    ap.add_argument("--qty-col", default="quantity")
    ap.add_argument("--revenue-col", default="revenue")
    ap.add_argument("--out", default=str(ROOT / "reports" / "evaluation_report.json"))
    args = ap.parse_args()

    q_table = np.load(args.q)
    log.info("Loaded Q-table %s from %s", q_table.shape, args.q)

    coef_path = Path(args.coefficients)
    if coef_path.exists():
        a_coef, b_coef = np.load(coef_path).tolist()
    else:
        a_coef, b_coef = 5.0, 1.5

    # Decide state mode from Q-table shape when "auto"
    mode = args.state_mode
    if mode == "auto":
        cfg_basic = EnvConfig()
        mode = "extended" if q_table.shape[0] == cfg_basic.num_price_bins * cfg_basic.num_competitor_bins else "basic"
        log.info("Auto-detected state_mode=%s from Q-table shape", mode)
    cfg = EnvConfig(a=float(a_coef), b=abs(float(b_coef)), seed=args.seed, state_mode=mode)
    env = DynamicPricingEnv(cfg)

    learned = run_policy(env, lambda s, _: int(np.argmax(q_table[s])), args.episodes, args.seed)
    random_pol = run_policy(env, lambda _s, rng: int(rng.integers(0, 3)),
                            args.episodes, args.seed + 1)

    def fixed_at(target: float):
        def policy(_s, _rng):
            if env.current_price < target - 1e-6: return 2
            if env.current_price > target + 1e-6: return 0
            return 1
        return policy

    floor = run_policy(env, fixed_at(cfg.price_floor),    args.episodes, args.seed + 2)
    ceil_ = run_policy(env, fixed_at(cfg.price_ceiling),  args.episodes, args.seed + 3)
    mid_  = run_policy(env, fixed_at((cfg.price_floor + cfg.price_ceiling) / 2),
                       args.episodes, args.seed + 4)

    def lift(a: float, b: float) -> float:
        return float((a - b) / b * 100) if abs(b) > 1e-9 else float("inf")

    report = {
        "evaluation": learned,
        "baselines": {"random": random_pol, "fixed_floor": floor,
                      "fixed_ceiling": ceil_, "fixed_midpoint": mid_},
        "lift_pct_vs_baseline": {
            "random":         lift(learned["mean_reward"], random_pol["mean_reward"]),
            "fixed_floor":    lift(learned["mean_reward"], floor["mean_reward"]),
            "fixed_ceiling":  lift(learned["mean_reward"], ceil_["mean_reward"]),
            "fixed_midpoint": lift(learned["mean_reward"], mid_["mean_reward"]),
        },
        "env_config": {**cfg.__dict__, "total_states": cfg.total_states},
    }

    # ── Held-out evaluation ──────────────────────────────────────────
    if args.holdout_csv and Path(args.holdout_csv).exists():
        df = pd.read_csv(args.holdout_csv)
        # date-sort if possible
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if date_col is not None:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.sort_values(date_col).reset_index(drop=True)
        cut = int(len(df) * (1 - args.holdout_frac))
        holdout = df.iloc[cut:].reset_index(drop=True)
        log.info("Held-out: %d rows (last %.0f%% of %s)", len(holdout), args.holdout_frac * 100, args.holdout_csv)
        qty_col = args.qty_col if args.qty_col in df.columns else None
        rev_col = args.revenue_col if args.revenue_col in df.columns else None
        price_col = args.price_col if args.price_col in df.columns else next(
            (c for c in df.columns if "price" in c.lower()), None)
        report["holdout_evaluation"] = holdout_evaluation(
            q_table, holdout, cfg, price_col, qty_col, rev_col
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    log.info("learned=%.3f random=%.3f mid=%.3f → wrote %s",
             learned["mean_reward"], random_pol["mean_reward"], mid_["mean_reward"], out)


if __name__ == "__main__":
    main()
