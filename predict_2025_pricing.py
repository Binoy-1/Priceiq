"""Generate weekly 2025 pricing recommendations from the trained policy.

Usage:
    python predict_2025_pricing.py --q models/q_table.npy \
        --weeks 52 --product "Wireless Headphones Pro" \
        --out reports/pricing_report.json

Walks the trained greedy policy week-by-week starting from the midpoint price,
samples demand from the fitted log-log model with seasonal multipliers, and
writes both pricing_report.json (summary) and recommendations CSV.
"""
from __future__ import annotations
import argparse
import json
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from environment import DynamicPricingEnv, EnvConfig
from utils.helpers import bin_price_to_state, predict_demand, ACTION_LABELS

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("predict_2025")

ROOT = Path(__file__).resolve().parent


def season_factor(week_idx: int) -> float:
    # Mild annual cycle + Q4 holiday lift
    s = 1.0 + 0.08 * np.sin(2 * np.pi * week_idx / 52)
    if week_idx >= 44:
        s *= 1.18
    return float(s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--q", default=str(ROOT / "models" / "q_table.npy"))
    ap.add_argument("--coefficients", default=str(ROOT / "models" / "coefficients.npy"))
    ap.add_argument("--weeks", type=int, default=52)
    ap.add_argument("--product", default="Wireless Headphones Pro")
    ap.add_argument("--start", default="2025-01-06", help="ISO date of week 1")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out", default=str(ROOT / "reports" / "pricing_report.json"))
    ap.add_argument("--csv", default=str(ROOT / "data" / "2025_retail_sample.csv"))
    args = ap.parse_args()

    q = np.load(args.q)
    log.info("Loaded Q-table %s", q.shape)

    coef_path = Path(args.coefficients)
    if coef_path.exists():
        a_coef, b_coef = np.load(coef_path).tolist()
    else:
        a_coef, b_coef = 5.0, 1.5
        log.warning("No coefficients at %s — using defaults", coef_path)

    cfg = EnvConfig(a=float(a_coef), b=abs(float(b_coef)), seed=args.seed)
    rng = np.random.default_rng(args.seed)

    price = (cfg.price_floor + cfg.price_ceiling) / 2.0
    start = pd.Timestamp(args.start)

    rows = []
    for w in range(args.weeks):
        state = bin_price_to_state(price, cfg.price_floor, cfg.price_ceiling, cfg.num_states)
        action = int(np.argmax(q[state]))
        # Apply action to price using env step size
        if action == 0:
            price = max(cfg.price_floor, price - cfg.step_size)
        elif action == 2:
            price = min(cfg.price_ceiling, price + cfg.step_size)
        sf = season_factor(w)
        demand = predict_demand(price, float(a_coef), abs(float(b_coef)), season=sf)
        demand += float(rng.normal(0.0, cfg.noise_std * 0.05))
        demand = max(0.0, demand)
        revenue = price * demand
        d = (start + pd.Timedelta(weeks=w)).date().isoformat()
        rows.append({
            "date": d,
            "week": w + 1,
            "quarter": f"Q{((w) // 13) + 1}",
            "product": args.product,
            "state": state,
            "action": ACTION_LABELS[action],
            "suggested_price": round(float(price), 2),
            "actual_demand": round(float(demand), 3),
            "actual_revenue": round(float(revenue), 2),
            "season_factor": round(sf, 3),
        })

    df = pd.DataFrame(rows)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    log.info("Wrote %d weekly recommendations → %s", len(df), args.csv)

    qsum = {}
    for q_label, sub in df.groupby("quarter"):
        qsum[q_label] = {
            "mean_price": float(sub["suggested_price"].mean()),
            "action_distribution": sub["action"].value_counts().to_dict(),
            "weeks": int(len(sub)),
        }

    report = {
        "product": args.product,
        "year": int(pd.Timestamp(args.start).year),
        "data_points": int(len(df)),
        "demand_model": {
            "equation": f"log(demand) = {a_coef:.4f} + {b_coef:.4f}*log(price)",
            "a": float(a_coef),
            "b": float(b_coef),
        },
        "action_summary": df["action"].value_counts().to_dict(),
        "quarterly_summary": qsum,
        "recommendations": df.to_dict(orient="records"),
        "totals": {
            "revenue": float(df["actual_revenue"].sum()),
            "mean_price": float(df["suggested_price"].mean()),
            "mean_demand": float(df["actual_demand"].mean()),
        },
        "csv": str(args.csv),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    log.info("Total revenue projected: $%.0f → %s", report["totals"]["revenue"], args.out)


if __name__ == "__main__":
    main()
