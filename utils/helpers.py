"""Shared business helpers + premium UI primitives."""
from __future__ import annotations
import numpy as np

ACTION_LABELS = {0: "LOWER", 1: "HOLD", 2: "RAISE"}
ACTION_COLORS = {"LOWER": "#6b6f4e", "HOLD": "#6b6259", "RAISE": "#cc785c"}
ACTION_DELTA = {0: -0.5, 1: 0.0, 2: +0.5}


def bin_price_to_state(price: float, floor: float = 1.0, ceiling: float = 20.0,
                       num_states: int = 9) -> int:
    price = max(floor, min(ceiling, float(price)))
    edges = np.linspace(floor, ceiling, num_states + 1)
    idx = int(np.digitize(price, edges) - 1)
    return max(0, min(num_states - 1, idx))


def predict_demand(price: float, a: float, b: float,
                   season: float = 1.0, demand_mult: float = 1.0) -> float:
    price = max(0.01, float(price))
    base = float(np.exp(a - b * np.log(price)))
    return max(0.0, base * season * demand_mult)


# ─── Premium UI primitives ──────────────────────────────────────────────

def hero(title: str, subtitle: str, badge: str | None = None) -> str:
    pill = f'<span class="pi-pill">{badge}</span>' if badge else ""
    return f"""
    <div class="pi-hero fade-in">
      {pill}
      <h1>{title}</h1>
      <p>{subtitle}</p>
    </div>
    """


def kpi_card(label: str, value: str, delta: str | None = None,
             neg: bool = False, icon: str = "◆",
             spark: list[float] | None = None) -> str:
    delta_html = ""
    if delta:
        cls = "delta neg" if neg else "delta"
        arrow = "▼" if neg else "▲"
        delta_html = f'<div class="{cls}">{arrow} {delta}</div>'

    spark_svg = ""
    if spark and len(spark) > 1:
        spark_svg = _sparkline_svg(spark)

    return f"""
    <div class="pi-kpi">
      <div class="top">
        <div class="lbl">{label}</div>
        <div class="icon">{icon}</div>
      </div>
      <div class="val">{value}</div>
      {delta_html}
      <div class="spark">{spark_svg}</div>
    </div>
    """


def _sparkline_svg(values: list[float], width: int = 180, height: int = 32,
                   stroke: str = "#cc785c") -> str:
    vals = list(values)
    n = len(vals)
    if n < 2:
        return ""
    vmin, vmax = min(vals), max(vals)
    rng = (vmax - vmin) or 1.0
    pts = []
    for i, v in enumerate(vals):
        x = i * (width / (n - 1))
        y = height - ((v - vmin) / rng) * height
        pts.append(f"{x:.1f},{y:.1f}")
    path = "M " + " L ".join(pts)
    area = path + f" L {width},{height} L 0,{height} Z"
    return f"""
    <svg width="100%" height="{height}" viewBox="0 0 {width} {height}" preserveAspectRatio="none">
      <defs>
        <linearGradient id="g{abs(hash(tuple(vals)))%99999}" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="{stroke}" stop-opacity="0.35"/>
          <stop offset="100%" stop-color="{stroke}" stop-opacity="0"/>
        </linearGradient>
      </defs>
      <path d="{area}" fill="url(#g{abs(hash(tuple(vals)))%99999})"/>
      <path d="{path}" fill="none" stroke="{stroke}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    """


def section(title: str, meta: str = "") -> str:
    return f'<div class="pi-section"><h2>{title}</h2><span class="meta">{meta}</span></div>'


def insight(text: str, label: str = "AI Insight") -> str:
    return f'<div class="pi-insight"><span class="ai">✦ {label}</span>{text}</div>'


def status_pill(label: str, kind: str = "ok") -> str:
    """kind: ok | warn | err"""
    dot = {"ok": "●", "warn": "●", "err": "●"}[kind]
    return f'<span class="pi-status {kind}">{dot} {label}</span>'
