"""Premium Plotly chart system — unified theme, gradients, smooth animations."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Brand palette — Anthropic-inspired warm editorial
BRAND   = "#cc785c"   # coral
BRAND_2 = "#a85a40"   # deep coral
ACCENT  = "#6b6f4e"   # olive
PINK    = "#b86b8a"
GREEN   = "#6b8f5e"
AMBER   = "#c89a3c"
TEXT    = "#1f1d1a"
DIM     = "#6b6259"
GRID    = "rgba(31,29,26,0.06)"

FONT = dict(family="Source Serif 4, Georgia, serif", size=12, color=TEXT)


def _layout(fig: go.Figure, title: str = "", height: int = 320,
            ytitle: str = "", xtitle: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(text=f"<span style='color:{TEXT};font-weight:600;font-size:14px'>{title}</span>",
                   x=0.0, xanchor="left", y=0.96, font=FONT),
        height=height,
        margin=dict(l=12, r=12, t=42, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=FONT,
        hoverlabel=dict(bgcolor="#faf6ef", bordercolor=BRAND,
                        font=dict(family="Source Serif 4", color=TEXT, size=12)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=DIM, size=11),
                    orientation="h", yanchor="bottom", y=-0.18, x=0),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                   tickfont=dict(color=DIM, size=11), title=xtitle,
                   showspikes=False),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                   tickfont=dict(color=DIM, size=11), title=ytitle),
        transition=dict(duration=400, easing="cubic-in-out"),
    )
    return fig


def line(x, y, title: str, y_label: str = "", color: str = BRAND,
         height: int = 300, fill: bool = True) -> go.Figure:
    fig = go.Figure()
    rgb = _hex_to_rgb(color)
    fig.add_trace(go.Scatter(
        x=list(x), y=list(y), mode="lines",
        line=dict(color=color, width=2.4, shape="spline", smoothing=0.6),
        fill="tozeroy" if fill else None,
        fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.10)",
        hovertemplate="<b>%{y:.3f}</b><extra></extra>",
    ))
    return _layout(fig, title, height, ytitle=y_label)


def bar(x, y, title: str, color: str = ACCENT, height: int = 300) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=list(x), y=list(y),
        marker=dict(color=color, line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>%{y:.3f}<extra></extra>",
    ))
    fig.update_traces(marker_line_width=0)
    return _layout(fig, title, height)


def heatmap(matrix: np.ndarray, x_labels, y_labels, title: str,
            height: int = 380) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=matrix, x=x_labels, y=y_labels,
        colorscale=[[0, "#faf6ef"], [0.4, "#f1d9cd"], [0.7, "#cc785c"], [1, "#a85a40"]],
        colorbar=dict(thickness=8, tickfont=dict(color=DIM, size=10),
                      outlinewidth=0),
        hovertemplate="state=%{y}<br>action=%{x}<br>Q=%{z:.4f}<extra></extra>",
    ))
    return _layout(fig, title, height)


def epsilon_curve(start: float, end: float, decay: float, episodes: int) -> go.Figure:
    eps = [start]
    for _ in range(episodes - 1):
        eps.append(max(end, eps[-1] * decay))
    return line(range(len(eps)), eps, "Epsilon Decay (Exploration → Exploitation)",
                "ε", color=PINK, height=260)


def demand_vs_price(a: float, b: float, floor: float = 1.0,
                    ceiling: float = 20.0, height: int = 340) -> go.Figure:
    prices = np.linspace(floor, ceiling, 100)
    demand = np.exp(a - b * np.log(np.maximum(prices, 0.01)))
    revenue = prices * demand
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prices, y=demand, mode="lines", name="Demand",
        line=dict(color=ACCENT, width=2.4, shape="spline"),
        fill="tozeroy", fillcolor="rgba(34,211,238,0.08)",
        hovertemplate="price=$%{x:.2f}<br>demand=%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=prices, y=revenue, mode="lines", name="Revenue",
        line=dict(color=BRAND, width=2.4, shape="spline"),
        yaxis="y2",
        hovertemplate="price=$%{x:.2f}<br>revenue=%{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        yaxis=dict(title="Demand", gridcolor=GRID, tickfont=dict(color=DIM)),
        yaxis2=dict(title="Revenue", overlaying="y", side="right",
                    gridcolor="rgba(0,0,0,0)", tickfont=dict(color=DIM)),
    )
    return _layout(fig, "Demand & Revenue Curves", height, xtitle="Price ($)")


def reward_distribution(mean: float, std: float, n: int = 2000) -> go.Figure:
    rng = np.random.default_rng(0)
    samples = rng.normal(mean, max(std, 1e-6), n)
    fig = go.Figure(go.Histogram(
        x=samples, nbinsx=42,
        marker=dict(color=BRAND, line=dict(width=0)),
        opacity=0.85,
        hovertemplate="%{x:.3f}<br>n=%{y}<extra></extra>",
    ))
    fig.add_vline(x=mean, line_color=ACCENT, line_width=2,
                  annotation_text=f"μ = {mean:.3f}",
                  annotation_font_color=ACCENT,
                  annotation_position="top right")
    return _layout(fig, "Reward Distribution (Monte Carlo, n=2000)", 300)


def quarterly_price(df_q: pd.DataFrame) -> go.Figure:
    palette = [BRAND, ACCENT, PINK, "#a78bfa"]
    fig = go.Figure(go.Bar(
        x=df_q["quarter"], y=df_q["mean_price"],
        marker=dict(color=palette[: len(df_q)], line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>$%{y:.2f}<extra></extra>",
    ))
    return _layout(fig, "Mean Suggested Price by Quarter", 300, ytitle="Price ($)")


def confidence_gauge(value: float, title: str = "Agent Confidence") -> go.Figure:
    """0–1 confidence gauge."""
    v = max(0.0, min(1.0, value)) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=v,
        number=dict(suffix="%", font=dict(color=TEXT, size=32, family="Source Serif 4")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=DIM, tickfont=dict(color=DIM, size=10)),
            bar=dict(color=BRAND, thickness=0.28),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, 40], color="rgba(244,63,94,0.15)"),
                dict(range=[40, 70], color="rgba(245,158,11,0.15)"),
                dict(range=[70, 100], color="rgba(16,185,129,0.15)"),
            ],
            threshold=dict(line=dict(color=ACCENT, width=3),
                           thickness=0.85, value=v),
        ),
    ))
    return _layout(fig, title, 240)


def action_distribution(counts: dict) -> go.Figure:
    """Donut of action distribution."""
    labels = list(counts.keys()); values = list(counts.values())
    colors = ["#22d3ee", "#a78bfa", "#f472b6"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.65,
        marker=dict(colors=colors[: len(labels)], line=dict(color="#0a0d14", width=2)),
        textfont=dict(color=TEXT, family="Source Serif 4"),
        hovertemplate="<b>%{label}</b><br>%{value} (%{percent})<extra></extra>",
    ))
    return _layout(fig, "Policy Action Distribution", 300)


def forecast_band(x, y, title: str = "Forecast", color: str = BRAND) -> go.Figure:
    """Line with confidence band."""
    arr = np.asarray(y, dtype=float)
    band = np.std(arr) * 0.6 + 0.001
    upper = arr + band; lower = arr - band
    rgb = _hex_to_rgb(color)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(x) + list(x)[::-1],
                             y=list(upper) + list(lower)[::-1],
                             fill="toself",
                             fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.12)",
                             line=dict(color="rgba(0,0,0,0)"),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=list(x), y=list(arr), mode="lines",
                             line=dict(color=color, width=2.4, shape="spline"),
                             name="Forecast",
                             hovertemplate="%{y:.2f}<extra></extra>"))
    return _layout(fig, title, 320)


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
