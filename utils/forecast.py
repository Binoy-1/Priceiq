"""Forecasting engine — Holt-Winters, ARIMA, linear/poly trend, optional Prophet.

All models return a unified ForecastResult so the UI can render any of them.
Pure-Python fallbacks ensure the app works even without Prophet/statsmodels.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd

ModelName = Literal["holt_winters", "arima", "linear", "poly", "naive_seasonal", "prophet"]


@dataclass
class ForecastResult:
    model: str
    history_dates: pd.Series
    history_values: pd.Series
    future_dates: pd.Series
    forecast: pd.Series
    lower: pd.Series
    upper: pd.Series
    metrics: dict[str, float]


def _infer_freq(dates: pd.Series) -> str:
    if len(dates) < 3:
        return "W"
    delta = (dates.iloc[-1] - dates.iloc[0]) / max(1, len(dates) - 1)
    days = delta.total_seconds() / 86400
    if days < 2:
        return "D"
    if days < 10:
        return "W"
    if days < 45:
        return "MS"
    return "QS"


def _future_index(last: pd.Timestamp, periods: int, freq: str) -> pd.DatetimeIndex:
    return pd.date_range(start=last, periods=periods + 1, freq=freq)[1:]


def _metrics(actual: np.ndarray, fitted: np.ndarray) -> dict[str, float]:
    err = actual - fitted
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / np.where(actual == 0, 1, actual)) * 100)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE_%": round(mape, 2)}


# ─── Models ─────────────────────────────────────────────────────────────

def _holt_winters(y: np.ndarray, horizon: int, season: int = 12) -> tuple[np.ndarray, np.ndarray]:
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        seasonal = "add" if len(y) >= 2 * season else None
        model = ExponentialSmoothing(y, trend="add", seasonal=seasonal,
                                     seasonal_periods=season if seasonal else None,
                                     initialization_method="estimated").fit()
        fc = np.asarray(model.forecast(horizon))
        fitted = np.asarray(model.fittedvalues)
        return fc, fitted
    except Exception:
        # Fallback: simple exponential smoothing + linear trend
        alpha = 0.3
        level = y[0]
        smoothed = [level]
        for v in y[1:]:
            level = alpha * v + (1 - alpha) * level
            smoothed.append(level)
        trend = (y[-1] - y[0]) / max(1, len(y) - 1)
        fc = np.array([smoothed[-1] + trend * (i + 1) for i in range(horizon)])
        return fc, np.array(smoothed)


def _arima(y: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(y, order=(2, 1, 1)).fit()
        fc = np.asarray(model.forecast(horizon))
        fitted = np.asarray(model.fittedvalues)
        return fc, fitted
    except Exception:
        return _holt_winters(y, horizon)


def _polyfit(y: np.ndarray, horizon: int, degree: int = 1) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(len(y))
    coef = np.polyfit(x, y, degree)
    poly = np.poly1d(coef)
    fitted = poly(x)
    fc = poly(np.arange(len(y), len(y) + horizon))
    return fc, fitted


def _naive_seasonal(y: np.ndarray, horizon: int, season: int = 12) -> tuple[np.ndarray, np.ndarray]:
    if len(y) < season:
        return _polyfit(y, horizon, 1)
    last = y[-season:]
    reps = int(np.ceil(horizon / season))
    fc = np.tile(last, reps)[:horizon]
    fitted = np.concatenate([y[:season], y[:-season]]) if len(y) >= 2 * season else y.copy()
    return fc, fitted


def _prophet(dates: pd.Series, y: np.ndarray, horizon: int,
             freq: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from prophet import Prophet
    df = pd.DataFrame({"ds": dates.values, "y": y})
    m = Prophet(interval_width=0.9, daily_seasonality=False, weekly_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=horizon, freq=freq)
    fc_df = m.predict(future)
    fc = fc_df["yhat"].tail(horizon).values
    lower = fc_df["yhat_lower"].tail(horizon).values
    upper = fc_df["yhat_upper"].tail(horizon).values
    fitted = fc_df["yhat"].head(len(y)).values
    return fc, lower, upper, fitted


# ─── Public API ─────────────────────────────────────────────────────────

def forecast(series: pd.DataFrame, value_col: str, horizon: int,
             model: ModelName = "holt_winters",
             confidence: float = 0.9) -> ForecastResult:
    """Forecast `value_col` `horizon` periods ahead. `series` needs a 'date' col."""
    s = series[["date", value_col]].dropna().sort_values("date").reset_index(drop=True)
    y = s[value_col].astype(float).values
    dates = s["date"]
    freq = _infer_freq(dates)
    season = {"D": 7, "W": 52, "MS": 12, "QS": 4}.get(freq, 12)

    if model == "prophet":
        try:
            fc, lower, upper, fitted = _prophet(dates, y, horizon, freq)
        except Exception:
            model = "holt_winters"

    if model in ("holt_winters",):
        fc, fitted = _holt_winters(y, horizon, season)
    elif model == "arima":
        fc, fitted = _arima(y, horizon)
    elif model == "linear":
        fc, fitted = _polyfit(y, horizon, 1)
    elif model == "poly":
        fc, fitted = _polyfit(y, horizon, 2)
    elif model == "naive_seasonal":
        fc, fitted = _naive_seasonal(y, horizon, season)

    if model != "prophet":
        # CI from residual std
        resid = y[-len(fitted):] - fitted[-len(y):] if len(fitted) >= len(y) else y[: len(fitted)] - fitted
        sigma = float(np.std(resid)) or float(np.std(y) * 0.1) or 1.0
        from scipy.stats import norm
        try:
            z = float(norm.ppf(0.5 + confidence / 2))
        except Exception:
            z = 1.645  # ~90%
        lower = fc - z * sigma
        upper = fc + z * sigma

    future_idx = _future_index(dates.iloc[-1], horizon, freq)
    fitted_aligned = np.asarray(fitted)[-len(y):] if len(fitted) >= len(y) else np.pad(
        fitted, (len(y) - len(fitted), 0), constant_values=y[0])
    return ForecastResult(
        model=model,
        history_dates=dates,
        history_values=s[value_col],
        future_dates=pd.Series(future_idx),
        forecast=pd.Series(fc),
        lower=pd.Series(lower),
        upper=pd.Series(upper),
        metrics=_metrics(y, fitted_aligned),
    )


def price_elasticity(price: np.ndarray, qty: np.ndarray) -> float:
    """Log-log slope; returns elasticity coefficient."""
    p = np.asarray(price, float)
    q = np.asarray(qty, float)
    mask = (p > 0) & (q > 0)
    if mask.sum() < 5:
        return float("nan")
    lp, lq = np.log(p[mask]), np.log(q[mask])
    slope, _ = np.polyfit(lp, lq, 1)
    return float(slope)


def revenue_optimum(price: np.ndarray, qty: np.ndarray,
                    grid: int = 60) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Sweep price grid using log-log demand fit; return (p*, expected_rev, grid, rev)."""
    p = np.asarray(price, float)
    q = np.asarray(qty, float)
    mask = (p > 0) & (q > 0)
    if mask.sum() < 5:
        return float("nan"), float("nan"), np.array([]), np.array([])
    lp, lq = np.log(p[mask]), np.log(q[mask])
    b, a = np.polyfit(lp, lq, 1)
    pgrid = np.linspace(p[mask].min() * 0.7, p[mask].max() * 1.3, grid)
    qhat = np.exp(a + b * np.log(pgrid))
    rev = pgrid * qhat
    i = int(np.argmax(rev))
    return float(pgrid[i]), float(rev[i]), pgrid, rev
