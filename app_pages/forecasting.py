"""Forecasting — dataset-aware (Holt-Winters, ARIMA, Prophet, linear/poly).

If a cleaned dataset exists in session_state, runs real forecasts on it.
Otherwise falls back to the bundled 2025 RL pricing report view.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.loaders import load_pricing_report, load_recommendations_df
from utils.helpers import hero, kpi_card, section, insight
from utils.charts import line, forecast_band, quarterly_price
from utils.forecast import forecast as run_forecast, price_elasticity, revenue_optimum


# ─── Dataset-driven mode ───────────────────────────────────────────────

def _plot_forecast(res, title: str, color: str = "#cc785c") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res.history_dates, y=res.history_values,
                             mode="lines", name="History",
                             line=dict(color="#3a3631", width=2)))
    fig.add_trace(go.Scatter(x=res.future_dates, y=res.upper, mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=res.future_dates, y=res.lower, mode="lines",
                             fill="tonexty", fillcolor="rgba(204,120,92,0.18)",
                             line=dict(width=0), name="Confidence"))
    fig.add_trace(go.Scatter(x=res.future_dates, y=res.forecast, mode="lines",
                             name="Forecast",
                             line=dict(color=color, width=2.6, dash="dot")))
    fig.update_layout(
        title=title, height=380,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color="#3a3631", size=12),
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", y=-0.2),
        xaxis=dict(gridcolor="#ece6dc"), yaxis=dict(gridcolor="#ece6dc"),
    )
    return fig


def _dataset_mode() -> None:
    df = st.session_state["active_dataset"]
    roles = st.session_state["active_roles"]
    ts = st.session_state.get("active_ts")

    if ts is None or ts.empty or "price" not in ts:
        st.error("Active dataset has no price/date series. Re-map columns in Data Studio.")
        return

    st.markdown(hero(
        "Forecasting Engine",
        f"Real forecasts on <strong>{st.session_state.get('dataset_name','dataset')}</strong>: "
        "Holt-Winters, ARIMA, Prophet, and trend models with confidence intervals.",
        badge="LIVE · DATASET",
    ), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        target = st.selectbox("Forecast target",
                              [c for c in ["price", "quantity", "revenue"] if c in ts.columns])
    with c2:
        model = st.selectbox("Model",
                             ["holt_winters", "arima", "prophet", "linear", "poly", "naive_seasonal"],
                             format_func=lambda x: {
                                 "holt_winters": "Holt-Winters",
                                 "arima": "ARIMA",
                                 "prophet": "Prophet",
                                 "linear": "Linear trend",
                                 "poly": "Polynomial",
                                 "naive_seasonal": "Naive seasonal",
                             }[x])
    with c3:
        horizon = st.slider("Periods ahead", 4, 104, 26)
    with c4:
        conf = st.slider("Confidence", 0.5, 0.99, 0.9, 0.05)

    try:
        res = run_forecast(ts, value_col=target, horizon=horizon,
                           model=model, confidence=conf)
    except Exception as e:
        st.error(f"Forecast failed: {e}")
        return

    # Metric cards
    last = float(ts[target].iloc[-1])
    pred_end = float(res.forecast.iloc[-1])
    delta = (pred_end - last) / max(1e-9, abs(last)) * 100
    kpis = [
        ("Last observed", f"{last:,.2f}", st.session_state.get("agg_freq", "W"),
         False, "◆"),
        ("Forecast (end of horizon)", f"{pred_end:,.2f}",
         f"{delta:+.1f}% vs last", delta < 0, "✦"),
        ("Model RMSE", f"{res.metrics['RMSE']:.3f}",
         f"MAPE {res.metrics['MAPE_%']}%", False, "≈"),
        ("MAE", f"{res.metrics['MAE']:.3f}", "in-sample fit", False, "✓"),
    ]
    cols = st.columns(4)
    for i, (l, v, d, neg, ic) in enumerate(kpis):
        with cols[i]:
            st.markdown(kpi_card(l, v, d, neg, icon=ic), unsafe_allow_html=True)

    st.markdown(section("Forecast",
                        f"{model.replace('_',' ').title()} · horizon {horizon}"),
                unsafe_allow_html=True)
    st.plotly_chart(_plot_forecast(res, f"{target.title()} forecast"),
                    use_container_width=True)

    # ── Seasonality decomposition ────────────────────────────────────
    with st.expander("Seasonality decomposition (trend / seasonal / residual)", expanded=False):
        try:
            from utils.seasonality import decompose
            decomp = decompose(ts.assign(date=ts["date"]), value_col=target)
            ck1, ck2, ck3 = st.columns(3)
            ck1.metric("Period (auto)", decomp.period)
            ck2.metric("Trend strength", f"{decomp.strength_trend:.2f}")
            ck3.metric("Seasonal strength", f"{decomp.strength_seasonal:.2f}")
            fg = go.Figure()
            fg.add_trace(go.Scatter(x=decomp.dates, y=decomp.observed, name="Observed",
                                    line=dict(color="#3a3631", width=1.6)))
            fg.add_trace(go.Scatter(x=decomp.dates, y=decomp.trend, name="Trend",
                                    line=dict(color="#cc785c", width=2.2)))
            fg.add_trace(go.Scatter(x=decomp.dates, y=decomp.seasonal, name="Seasonal",
                                    line=dict(color="#6b6f4e", width=1.4, dash="dot")))
            fg.add_trace(go.Scatter(x=decomp.dates, y=decomp.residual, name="Residual",
                                    line=dict(color="#a09a8e", width=1.0)))
            fg.update_layout(
                title=f"{target.title()} · decomposition ({decomp.method})",
                height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#3a3631"),
                margin=dict(l=20, r=20, t=50, b=30),
                legend=dict(orientation="h", y=-0.2),
                xaxis=dict(gridcolor="#ece6dc"), yaxis=dict(gridcolor="#ece6dc"),
            )
            st.plotly_chart(fg, use_container_width=True)
            st.caption(
                f"Decomposition method: {decomp.method}. Strength scores in [0,1] — "
                "values near 1 indicate that component explains most of the variance."
            )
        except Exception as ex:
            st.warning(f"Could not decompose series: {ex}")

    # Elasticity / optimum
    if "price" in ts and "quantity" in ts:
        st.markdown(section("Price elasticity & revenue optimum",
                            "Log-log demand fit"), unsafe_allow_html=True)
        e = price_elasticity(ts["price"].values, ts["quantity"].values)
        p_opt, r_opt, pgrid, rev = revenue_optimum(ts["price"].values, ts["quantity"].values)
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pgrid, y=rev, mode="lines",
                                     line=dict(color="#cc785c", width=2.4)))
            if not np.isnan(p_opt):
                fig.add_vline(x=p_opt, line_dash="dot", line_color="#3a3631")
            fig.update_layout(title="Expected revenue vs price",
                              height=360, plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(family="Inter", color="#3a3631"),
                              xaxis_title="Price", yaxis_title="Revenue",
                              margin=dict(l=20, r=20, t=50, b=30))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            msg = (f"Estimated price elasticity: <strong>{e:.2f}</strong>. "
                   f"Revenue-maximizing price ≈ <strong>{p_opt:,.2f}</strong> "
                   f"(expected revenue {r_opt:,.0f} per period).") if not np.isnan(p_opt) else (
                "Insufficient paired price/quantity data to compute elasticity.")
            st.markdown(insight(msg, label="Strategy"), unsafe_allow_html=True)

    # Forecast table + exports
    out = pd.DataFrame({
        "date": res.future_dates.values,
        f"{target}_forecast": res.forecast.values,
        "lower": res.lower.values,
        "upper": res.upper.values,
    })
    st.markdown(section("Forecast table", f"{len(out)} predicted periods"),
                unsafe_allow_html=True)
    st.dataframe(out, use_container_width=True, hide_index=True)

    from utils.exports import df_to_csv_bytes, df_to_xlsx_bytes, report_to_pdf_bytes
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("⬇  Forecast CSV", df_to_csv_bytes(out),
                           file_name=f"{target}_forecast.csv", mime="text/csv",
                           use_container_width=True)
    with c2:
        st.download_button("⬇  Forecast XLSX",
                           df_to_xlsx_bytes({"forecast": out, "history": ts}),
                           file_name=f"{target}_forecast.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    with c3:
        pdf = report_to_pdf_bytes(
            f"{target.title()} Forecast Report",
            sections=[
                ("Model", f"{model} · horizon {horizon} · confidence {conf:.0%}"),
                ("Accuracy", f"MAE {res.metrics['MAE']:.3f} · RMSE {res.metrics['RMSE']:.3f} · "
                             f"MAPE {res.metrics['MAPE_%']}%"),
                ("Outlook", f"Forecast moves from {last:.2f} to {pred_end:.2f} "
                            f"({delta:+.1f}%) by end of horizon."),
            ],
            tables={"Forecast": out.head(40)},
        )
        st.download_button("⬇  Forecast PDF", pdf,
                           file_name=f"{target}_forecast.pdf",
                           mime="application/pdf",
                           use_container_width=True)


# ─── RL-report mode (legacy) ───────────────────────────────────────────

def _rl_mode() -> None:
    st.markdown(hero(
        "2025 Pricing Forecast",
        "Bundled RL policy view — upload a dataset in Data Studio for live forecasting.",
        badge="FORECAST · RL POLICY",
    ), unsafe_allow_html=True)

    rep = load_pricing_report()
    df = load_recommendations_df()
    if rep is None or df.empty:
        st.warning("No forecast available. Upload a dataset in Data Studio to begin.")
        return

    total_rev = float(df["actual_revenue"].sum())
    mean_price = float(df["suggested_price"].mean())
    holiday = df[df["date"].dt.month.isin([11, 12])]
    holiday_avg = float(holiday["suggested_price"].mean()) if not holiday.empty else 0.0
    lower = (df["action"] == "LOWER").sum() if "action" in df else 0
    raise_ = (df["action"] == "RAISE").sum() if "action" in df else 0

    cols = st.columns(4)
    cards = [
        ("Projected revenue", f"${total_rev:,.0f}", "FY 2025", False, "$"),
        ("Mean price", f"${mean_price:.2f}", "weekly avg", False, "◆"),
        ("Holiday avg", f"${holiday_avg:.2f}",
            f"{(holiday_avg/mean_price-1)*100:+.1f}% vs annual" if holiday_avg else "—",
            False, "✦"),
        ("LOWER / RAISE", f"{lower} / {raise_}", "weekly actions", False, "⇅"),
    ]
    for i, (l, v, d, neg, ic) in enumerate(cards):
        with cols[i]:
            st.markdown(kpi_card(l, v, d, neg, icon=ic), unsafe_allow_html=True)

    quarters = ["All"] + sorted(df["quarter"].unique().tolist())
    pick = st.selectbox("Quarter", quarters, label_visibility="collapsed")
    fdf = df if pick == "All" else df[df["quarter"] == pick]
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(forecast_band(fdf["date"], fdf["suggested_price"],
                                      "Weekly Suggested Price (with confidence band)",
                                      color="#cc785c"),
                        use_container_width=True)
    with c2:
        st.plotly_chart(line(fdf["date"], fdf["actual_revenue"],
                             "Weekly Revenue", "Revenue ($)", color="#6b6f4e"),
                        use_container_width=True)

    qsum = rep.get("quarterly_summary", {})
    if qsum:
        qdf = pd.DataFrame([{"quarter": k, "mean_price": v["mean_price"], "weeks": v["weeks"]}
                            for k, v in qsum.items()])
        st.plotly_chart(quarterly_price(qdf), use_container_width=True)


# ─── Entry ─────────────────────────────────────────────────────────────

def render() -> None:
    if st.session_state.get("active_dataset") is not None:
        _dataset_mode()
    else:
        _rl_mode()
