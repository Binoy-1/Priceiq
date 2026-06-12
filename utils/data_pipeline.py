"""Real-dataset ingestion: load any CSV/XLSX/JSON, infer schema, clean, score.

Designed to handle arbitrary user-uploaded pricing data — ecommerce, retail,
stock, dynamic pricing — with zero hardcoded assumptions about column names.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from io import BytesIO
from typing import Any
import re
import json
import numpy as np
import pandas as pd

# Column-role detection patterns (case-insensitive substrings)
ROLE_PATTERNS: dict[str, list[str]] = {
    "price":    ["price", "unit_price", "unitprice", "amount", "cost", "rate", "fare", "value", "close", "open"],
    "date":     ["date", "time", "timestamp", "day", "week", "month", "period"],
    "product":  ["product", "sku", "item", "ticker", "symbol", "category", "name", "title"],
    "quantity": ["qty", "quantity", "units", "demand", "volume", "sold", "count", "orders"],
    "revenue":  ["revenue", "sales", "turnover", "gmv", "income", "total"],
}

CURRENCY_RE = re.compile(r"[\$€£¥₹₽]|usd|eur|gbp|jpy|inr|aud|cad|chf", re.I)


@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    nulls_filled: int = 0
    rows_dropped_nulls: int = 0
    outliers_capped: int = 0
    currency_normalized: int = 0
    dates_parsed: int = 0
    quality_score: float = 0.0
    issues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─── Loading ────────────────────────────────────────────────────────────

def read_uploaded(file) -> pd.DataFrame:
    """Load a Streamlit UploadedFile (CSV / XLSX / JSON) into a DataFrame."""
    name = getattr(file, "name", "uploaded").lower()
    raw = file.read() if hasattr(file, "read") else file
    bio = BytesIO(raw if isinstance(raw, (bytes, bytearray)) else raw.encode())

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(bio)
    if name.endswith(".json"):
        bio.seek(0)
        try:
            return pd.read_json(bio)
        except ValueError:
            bio.seek(0)
            data = json.load(bio)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, list):
                        return pd.DataFrame(v)
            return pd.json_normalize(data)
    # default: CSV — try a few separators
    for sep in [",", ";", "\t", "|"]:
        bio.seek(0)
        try:
            df = pd.read_csv(bio, sep=sep, engine="python")
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    bio.seek(0)
    return pd.read_csv(bio)


# ─── Schema detection ───────────────────────────────────────────────────

def detect_roles(df: pd.DataFrame) -> dict[str, str | None]:
    """Map roles → best-matching column name (or None)."""
    cols = list(df.columns)
    lower = {c: str(c).lower().strip() for c in cols}
    used: set[str] = set()
    out: dict[str, str | None] = {}

    for role, patterns in ROLE_PATTERNS.items():
        best = None
        for pat in patterns:
            for c in cols:
                if c in used:
                    continue
                if pat in lower[c]:
                    best = c
                    break
            if best:
                break
        # numeric fallback for price/quantity/revenue
        if best is None and role in {"price", "quantity", "revenue"}:
            for c in cols:
                if c in used:
                    continue
                if pd.api.types.is_numeric_dtype(df[c]):
                    best = c
                    break
        # date fallback: try parsing
        if best is None and role == "date":
            for c in cols:
                if c in used:
                    continue
                try:
                    parsed = pd.to_datetime(df[c], errors="coerce")
                    if parsed.notna().sum() > len(df) * 0.5:
                        best = c
                        break
                except Exception:
                    pass
        if best:
            used.add(best)
        out[role] = best
    return out


# ─── Cleaning ───────────────────────────────────────────────────────────

def _to_numeric_currency(s: pd.Series) -> tuple[pd.Series, int]:
    """Strip currency symbols and parse numbers. Returns (series, n_normalized)."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce"), 0
    converted = s.astype(str).str.replace(CURRENCY_RE, "", regex=True)
    converted = converted.str.replace(r"[,\s]", "", regex=True)
    out = pd.to_numeric(converted, errors="coerce")
    n = int((out.notna() & s.notna()).sum())
    return out, n


def clean_dataset(df: pd.DataFrame, roles: dict[str, str | None],
                  outlier_strategy: str = "cap") -> tuple[pd.DataFrame, CleaningReport]:
    """Run full cleaning pipeline. outlier_strategy: 'cap' | 'drop' | 'keep'."""
    rep = CleaningReport(rows_in=len(df))
    df = df.copy()

    # 1. Drop fully-empty columns
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        rep.notes.append(f"Dropped {len(empty_cols)} empty column(s).")

    # 2. Parse date column
    date_col = roles.get("date")
    if date_col and date_col in df.columns:
        parsed = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
        rep.dates_parsed = int(parsed.notna().sum())
        df[date_col] = parsed
        before = len(df)
        df = df.dropna(subset=[date_col])
        rep.rows_dropped_nulls += before - len(df)

    # 3. Currency-normalize numeric-ish columns
    for role in ("price", "revenue", "quantity"):
        col = roles.get(role)
        if col and col in df.columns:
            new, n = _to_numeric_currency(df[col])
            df[col] = new
            rep.currency_normalized += n

    # 4. Duplicate removal
    before = len(df)
    df = df.drop_duplicates()
    rep.duplicates_removed = before - len(df)

    # 5. Null handling — fill numeric with median, drop critical-null rows
    price_col = roles.get("price")
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            n_null = int(df[c].isna().sum())
            if n_null:
                df[c] = df[c].fillna(df[c].median())
                rep.nulls_filled += n_null
    if price_col and price_col in df.columns:
        before = len(df)
        df = df.dropna(subset=[price_col])
        rep.rows_dropped_nulls += before - len(df)

    # 6. Outlier handling on price (IQR fences)
    if price_col and price_col in df.columns and outlier_strategy != "keep":
        s = df[price_col]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        mask = (s < lo) | (s > hi)
        n_out = int(mask.sum())
        if n_out:
            if outlier_strategy == "cap":
                df.loc[s < lo, price_col] = lo
                df.loc[s > hi, price_col] = hi
                rep.outliers_capped = n_out
                rep.notes.append(f"Capped {n_out} price outlier(s) outside IQR×3 fences.")
            elif outlier_strategy == "drop":
                df = df[~mask]
                rep.rows_dropped_nulls += n_out

    # 7. Sort by date
    if date_col and date_col in df.columns:
        df = df.sort_values(date_col).reset_index(drop=True)

    # 8. Issues / quality score
    rep.rows_out = len(df)
    if rep.rows_in > 0:
        retention = rep.rows_out / rep.rows_in
        density = 1 - (df.isna().sum().sum() / max(1, df.size))
        # Schema completeness — reward mapped roles
        mapped = sum(1 for v in roles.values() if v)
        schema = mapped / len(ROLE_PATTERNS)
        rep.quality_score = round(100 * (0.4 * retention + 0.4 * density + 0.2 * schema), 1)

    if not roles.get("price"):
        rep.issues.append("No price column detected — forecasting will be unavailable.")
    if not roles.get("date"):
        rep.issues.append("No date column detected — time-series features disabled.")
    if rep.rows_out < 30:
        rep.issues.append(f"Only {rep.rows_out} rows after cleaning — forecasts will be low-confidence.")

    return df, rep


# ─── Aggregation helpers ────────────────────────────────────────────────

def aggregate_timeseries(df: pd.DataFrame, roles: dict[str, str | None],
                         freq: str = "W") -> pd.DataFrame:
    """Aggregate the cleaned frame to a regular frequency time series."""
    date_col = roles.get("date")
    price_col = roles.get("price")
    qty_col = roles.get("quantity")
    rev_col = roles.get("revenue")
    if not date_col or not price_col:
        return pd.DataFrame()

    g = df.set_index(date_col).sort_index()
    agg: dict[str, Any] = {price_col: "mean"}
    if qty_col and qty_col in g:
        agg[qty_col] = "sum"
    if rev_col and rev_col in g:
        agg[rev_col] = "sum"

    out = g.resample(freq).agg(agg).dropna(subset=[price_col]).reset_index()
    out = out.rename(columns={date_col: "date", price_col: "price",
                              qty_col or "_qty": "quantity",
                              rev_col or "_rev": "revenue"})
    if "revenue" not in out and "quantity" in out:
        out["revenue"] = out["price"] * out["quantity"]
    return out
