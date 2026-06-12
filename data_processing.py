"""Fit a log-log demand model from a CSV and persist coefficients.

Usage:
    python data_processing.py --input data/2025_retail_sample.csv \
        --price-col suggested_price --quantity-col actual_demand \
        --out models/coefficients.npy

If --quantity-col is omitted, the script will derive demand from
revenue / price when both are present. Saves a numpy array
[a, b] where  log(demand) = a + b * log(price).

This is the single source of truth for the demand model used by
environment.py (EnvConfig.a, EnvConfig.b) and the dashboard.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("data_processing")

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "models" / "coefficients.npy"
PROVENANCE_OUT = ROOT / "models" / "coefficients_provenance.json"


def fit_loglog(price: np.ndarray, qty: np.ndarray) -> tuple[float, float, dict]:
    p = np.asarray(price, dtype=float)
    q = np.asarray(qty, dtype=float)
    mask = (p > 0) & (q > 0) & np.isfinite(p) & np.isfinite(q)
    if mask.sum() < 5:
        raise ValueError(
            f"Need at least 5 paired (price>0, quantity>0) rows; got {int(mask.sum())}."
        )
    lp, lq = np.log(p[mask]), np.log(q[mask])
    # log(q) = a + b * log(p)   ->  np.polyfit returns [b, a]
    b, a = np.polyfit(lp, lq, 1)
    fitted = a + b * lp
    ss_res = float(np.sum((lq - fitted) ** 2))
    ss_tot = float(np.sum((lq - lq.mean()) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    diag = {
        "n": int(mask.sum()),
        "a_intercept": float(a),
        "b_slope_elasticity": float(b),
        "r_squared": float(r2),
        "price_range": [float(p[mask].min()), float(p[mask].max())],
        "qty_range": [float(q[mask].min()), float(q[mask].max())],
    }
    return float(a), float(b), diag


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Path to CSV/XLSX with price & quantity")
    ap.add_argument("--price-col", default=None)
    ap.add_argument("--quantity-col", default=None)
    ap.add_argument("--revenue-col", default=None,
                    help="If quantity-col missing, derive qty = revenue / price")
    ap.add_argument("--by", default=None,
                    help="Optional grouping column (product/SKU/segment). "
                         "Writes per-group elasticity to coefficients_by_product.json.")
    ap.add_argument("--min-rows", type=int, default=10,
                    help="Skip groups with fewer than this many rows.")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    in_path = Path(args.input)
    if in_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(in_path)
    else:
        df = pd.read_csv(in_path)
    log.info("Loaded %d rows, columns=%s", len(df), list(df.columns))

    # Auto-detect if not specified
    cols_lower = {c.lower(): c for c in df.columns}
    price_col = args.price_col or next(
        (cols_lower[k] for k in cols_lower if "price" in k), None)
    qty_col = args.quantity_col or next(
        (cols_lower[k] for k in cols_lower if k in ("demand", "quantity", "qty", "actual_demand", "units")), None)
    rev_col = args.revenue_col or next(
        (cols_lower[k] for k in cols_lower if "revenue" in k or "sales" in k), None)

    if price_col is None:
        raise SystemExit("Could not find a price column. Pass --price-col explicitly.")
    log.info("price_col=%s | quantity_col=%s | revenue_col=%s", price_col, qty_col, rev_col)

    if qty_col is not None:
        qty = df[qty_col].astype(float).values
    elif rev_col is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            qty = (df[rev_col].astype(float) / df[price_col].astype(float)).values
        log.info("Derived quantity from revenue / price.")
    else:
        raise SystemExit("Provide --quantity-col or --revenue-col.")

    a, b, diag = fit_loglog(df[price_col].astype(float).values, qty)
    log.info("Fit: a=%.4f  b=%.4f  R²=%.3f  (n=%d)",
             a, b, diag["r_squared"], diag["n"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.array([a, b], dtype=np.float64))
    log.info("Saved coefficients → %s", out)

    PROVENANCE_OUT.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE_OUT.write_text(json.dumps({
        "source_file": str(in_path),
        "price_col": price_col, "quantity_col": qty_col, "revenue_col": rev_col,
        **diag,
    }, indent=2))
    log.info("Wrote provenance → %s", PROVENANCE_OUT)

    # ── Per-group elasticity (optional) ─────────────────────────────────
    if args.by:
        if args.by not in df.columns:
            log.warning("--by %s not in columns; skipping per-group fit", args.by)
        else:
            by_path = PROVENANCE_OUT.parent / "coefficients_by_product.json"
            results: dict[str, dict] = {}
            for name, sub in df.groupby(args.by):
                if len(sub) < args.min_rows:
                    continue
                if qty_col is not None:
                    sub_q = sub[qty_col].astype(float).values
                else:
                    sub_q = (sub[rev_col].astype(float) / sub[price_col].astype(float)).values
                try:
                    ga, gb, gdiag = fit_loglog(sub[price_col].astype(float).values, sub_q)
                    results[str(name)] = {"a": ga, "b": gb, **gdiag}
                except Exception as e:
                    log.warning("[%s] fit failed: %s", name, e)
            by_path.write_text(json.dumps(results, indent=2))
            log.info("Per-group elasticity for %d groups → %s", len(results), by_path)


if __name__ == "__main__":
    main()
