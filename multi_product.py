"""Multi-product RL training pipeline.

For each unique product in a dataset, fits its own log-log demand
coefficients and trains a per-product Q-table. Outputs:

  models/multi/coefficients_by_product.json   {product: {a, b, n, r2}}
  models/multi/q_tables.npz                    keys=products, values=Q-tables
  models/multi/summary.json                    per-product training stats

Usage:
    python multi_product.py --input data/online_retail_ii.csv \
        --product-col StockCode --price-col Price --quantity-col Quantity \
        --episodes 600 --state-mode extended --top 25
"""
from __future__ import annotations
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from agent import QLearningAgent, AgentConfig
from environment import DynamicPricingEnv, EnvConfig
from data_processing import fit_loglog

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("multi_product")
ROOT = Path(__file__).resolve().parent


def train_one(a_coef: float, b_coef: float, episodes: int, state_mode: str,
              cost: float, seed: int) -> tuple[np.ndarray, float]:
    cfg_e = EnvConfig(a=float(a_coef), b=abs(float(b_coef)), cost=float(cost),
                      seed=seed, state_mode=state_mode)
    cfg_a = AgentConfig(num_states=cfg_e.total_states, seed=seed)
    agent = QLearningAgent(cfg_a); env = DynamicPricingEnv(cfg_e)
    rewards = []
    for _ in range(episodes):
        state = env.reset(); total = 0.0; done = False
        while not done:
            action = agent.choose_action(int(state))
            next_state, reward, done, _ = env.step(action)
            agent.update(int(state), int(action), float(reward), int(next_state), bool(done))
            state = next_state; total += float(reward)
        rewards.append(total)
    return agent.q_table, float(np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--product-col", default="product")
    ap.add_argument("--price-col", default="price")
    ap.add_argument("--quantity-col", default="quantity")
    ap.add_argument("--revenue-col", default=None)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--state-mode", choices=["basic", "extended"], default="basic")
    ap.add_argument("--top", type=int, default=20, help="Train on top-N products by row count.")
    ap.add_argument("--min-rows", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=str(ROOT / "models" / "multi"))
    args = ap.parse_args()

    p = Path(args.input)
    df = pd.read_excel(p) if p.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(p)
    log.info("Loaded %d rows from %s", len(df), p)

    for c in (args.product_col, args.price_col):
        if c not in df.columns:
            raise SystemExit(f"Missing column: {c}. Have: {list(df.columns)}")
    if args.quantity_col not in df.columns and args.revenue_col is None:
        raise SystemExit("Provide --quantity-col or --revenue-col.")

    if args.quantity_col not in df.columns:
        df["__qty__"] = df[args.revenue_col].astype(float) / df[args.price_col].astype(float).clip(1e-9)
        qcol = "__qty__"
    else:
        qcol = args.quantity_col

    counts = df.groupby(args.product_col).size().sort_values(ascending=False)
    products = counts[counts >= args.min_rows].head(args.top).index.tolist()
    log.info("Training %d products (min_rows=%d, top=%d)", len(products), args.min_rows, args.top)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    coeffs: dict[str, dict] = {}
    qtables: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}

    t0 = time.time()
    for prod in products:
        sub = df[df[args.product_col] == prod]
        try:
            a_coef, b_coef, diag = fit_loglog(sub[args.price_col].values, sub[qcol].values)
        except Exception as e:
            log.warning("[%s] fit failed: %s — skipping", prod, e)
            continue
        cost = float(0.4 * sub[args.price_col].astype(float).mean())
        qtable, final_mean = train_one(a_coef, b_coef, args.episodes,
                                       args.state_mode, cost, args.seed)
        coeffs[str(prod)] = {**diag, "cost_used": cost}
        qtables[str(prod)] = qtable
        summary[str(prod)] = {
            "rows": int(len(sub)),
            "a": float(a_coef), "b": float(b_coef), "r2": diag["r_squared"],
            "cost": cost,
            "final_mean_reward_last_100": final_mean,
            "q_table_shape": list(qtable.shape),
        }
        log.info("[%-12s] n=%4d  a=%6.3f b=%6.3f  cost=%5.2f  reward=%.3f",
                 str(prod)[:12], len(sub), a_coef, b_coef, cost, final_mean)

    (outdir / "coefficients_by_product.json").write_text(json.dumps(coeffs, indent=2))
    np.savez_compressed(outdir / "q_tables.npz", **qtables)
    (outdir / "summary.json").write_text(json.dumps({
        "trained_products": list(qtables.keys()),
        "episodes": args.episodes,
        "state_mode": args.state_mode,
        "elapsed_seconds": round(time.time() - t0, 2),
        "per_product": summary,
    }, indent=2))
    log.info("Done. Trained %d products in %.1fs → %s", len(qtables), time.time() - t0, outdir)


if __name__ == "__main__":
    main()
