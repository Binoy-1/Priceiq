"""Seasonality decomposition utilities (trend + seasonal + residual).

Uses statsmodels.seasonal_decompose when available; otherwise falls back to a
moving-average trend + per-period seasonal averages so the project remains
runnable without statsmodels.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Decomposition:
    dates: pd.Series
    observed: pd.Series
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    period: int
    method: str
    strength_trend: float
    strength_seasonal: float


def _infer_period(dates: pd.Series, n: int) -> int:
    if len(dates) < 3:
        return max(2, min(12, n // 2))
    delta = (dates.iloc[-1] - dates.iloc[0]) / max(1, len(dates) - 1)
    days = delta.total_seconds() / 86400
    if days < 2:    return min(7,  max(2, n // 2))
    if days < 10:   return min(52, max(4, n // 2))
    if days < 45:   return min(12, max(4, n // 2))
    return min(4,  max(2, n // 2))


def decompose(ts: pd.DataFrame, value_col: str, period: int | None = None) -> Decomposition:
    s = ts[["date", value_col]].dropna().sort_values("date").reset_index(drop=True)
    y = s[value_col].astype(float).values
    n = len(y)
    if n < 6:
        raise ValueError(f"Need at least 6 observations to decompose; got {n}.")
    p = period or _infer_period(s["date"], n)
    p = max(2, min(p, n // 2))

    method = "statsmodels.seasonal_decompose"
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        res = seasonal_decompose(pd.Series(y, index=s["date"]),
                                 model="additive", period=p, extrapolate_trend="freq")
        trend, seasonal, resid = res.trend.values, res.seasonal.values, res.resid.values
    except Exception:
        method = "fallback_moving_average"
        # Centered moving average for trend
        if p % 2 == 0:
            k = p // 2
            ma = np.convolve(y, np.ones(p) / p, mode="same")
            ma = (ma + np.roll(ma, -1)) / 2
            trend = ma
        else:
            trend = np.convolve(y, np.ones(p) / p, mode="same")
        detr = y - trend
        seasonal = np.zeros_like(y)
        for i in range(p):
            idx = np.arange(i, n, p)
            seasonal[idx] = np.nanmean(detr[idx])
        seasonal -= np.nanmean(seasonal)
        resid = y - trend - seasonal

    var_resid    = float(np.nanvar(resid))
    var_detrend  = float(np.nanvar(y - trend))         or 1e-12
    var_deseason = float(np.nanvar(y - seasonal))      or 1e-12
    strength_trend    = max(0.0, 1.0 - var_resid / var_deseason)
    strength_seasonal = max(0.0, 1.0 - var_resid / var_detrend)

    return Decomposition(
        dates=s["date"], observed=pd.Series(y),
        trend=pd.Series(trend), seasonal=pd.Series(seasonal),
        residual=pd.Series(resid), period=p, method=method,
        strength_trend=float(strength_trend),
        strength_seasonal=float(strength_seasonal),
    )
