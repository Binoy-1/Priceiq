"""Live Pricing Simulator — premium what-if workbench."""
from __future__ import annotations
import numpy as np
import streamlit as st

from utils.loaders import load_q_table, load_coefficients
from utils.helpers import (hero, bin_price_to_state, predict_demand,
                           ACTION_LABELS, ACTION_DELTA, insight, section)
from utils.charts import demand_vs_price, confidence_gauge, bar


def render() -> None:
    st.markdown(hero(
        "Live Pricing Simulator",
        "Probe the trained policy with any market scenario. The agent picks the "
        "action with the highest expected reward and explains why.",
        badge="WHAT-IF · INTERACTIVE",
    ), unsafe_allow_html=True)

    q = load_q_table(); coef = load_coefficients()
    if q is None or coef is None:
        st.error("Missing q_table.npy or coefficients.npy in /models.")
        return
    a, b = float(coef[0]), float(coef[1])

    left, right = st.columns([1, 1.3], gap="large")

    with left:
        st.markdown('<div class="pi-card"><h3>Market State</h3>', unsafe_allow_html=True)
        current_price    = st.slider("Current price ($)", 1.0, 20.0, 12.5, 0.1)
        competitor_price = st.slider("Competitor price ($)", 1.0, 20.0, 13.0, 0.1)
        season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Holiday (Q4)"])
        demand_mult = st.slider("Demand shock multiplier", 0.5, 2.0, 1.0, 0.05)
        inventory = st.slider("Inventory (units)", 0, 5000, 1200, 50)
        st.markdown('</div>', unsafe_allow_html=True)

    season_factor = {"Spring": 1.0, "Summer": 1.05, "Autumn": 0.95, "Holiday (Q4)": 1.25}[season]
    state = bin_price_to_state(current_price)
    q_row = q[state].astype(float).copy()

    # ── Competitor-aware adjustment ───────────────────────────────────
    # The tabular Q-table is trained on price bins only. To make the
    # competitor slider actually steer the recommendation (instead of being
    # cosmetic), we add a market-share term to each candidate action's
    # value: lowering price closes a positive gap (we are pricier), raising
    # exploits a negative gap (we are cheaper). The weight is small so a
    # confidently-trained Q-row still dominates.
    competitor_gap = current_price - competitor_price  # >0 means we are pricier
    q_spread = float(np.max(q_row) - np.min(q_row)) or 1e-3
    w = 0.35 * q_spread  # competitor influence capped at ~35% of Q spread
    # Per-action share signal in [-1, 1]
    share_signal = np.array([
        +np.tanh(competitor_gap / 2.0),   # LOWER helps when we're pricier
        -0.25 * np.tanh(abs(competitor_gap) / 2.0),  # HOLD slightly penalised when far apart
        -np.tanh(competitor_gap / 2.0),   # RAISE helps when we're cheaper
    ])
    q_adjusted = q_row + w * share_signal

    action = int(np.argmax(q_adjusted))
    z = q_adjusted - q_adjusted.max()
    confidence = float(np.exp(z[action]) / np.sum(np.exp(z)))
    suggested_price = round(max(1.0, min(20.0, current_price + ACTION_DELTA[action])), 2)
    demand = predict_demand(suggested_price, a, b, season_factor, demand_mult)
    expected_revenue = round(suggested_price * demand, 2)
    competitor_gap = current_price - competitor_price

    with right:
        st.markdown('<div class="pi-card"><h3>AI Recommendation</h3>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Action", ACTION_LABELS[action])
        c2.metric("Suggested price", f"${suggested_price:.2f}", f"{ACTION_DELTA[action]:+.2f}")
        c3.metric("Expected revenue", f"${expected_revenue:,.2f}")
        c4, c5, c6 = st.columns(3)
        c4.metric("State bin", f"#{state}/8")
        c5.metric("Predicted demand", f"{demand:.2f}")
        c6.metric("vs Competitor", f"${competitor_gap:+.2f}")
        st.plotly_chart(confidence_gauge(confidence), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Reasoning
    reasons = []
    if action == 0:
        reasons.append("Q-value for <strong>LOWER</strong> dominates — elasticity favors more units.")
    elif action == 2:
        reasons.append("Q-value for <strong>RAISE</strong> dominates — demand is inelastic enough to gain margin.")
    else:
        reasons.append("Q-values are tightly clustered — <strong>HOLD</strong> preserves expected reward.")
    if competitor_gap > 0.5:
        reasons.append(f"Priced <strong>${competitor_gap:.2f} above</strong> competitor — switching risk.")
    elif competitor_gap < -0.5:
        reasons.append(f"Priced <strong>${-competitor_gap:.2f} below</strong> competitor — capture share.")
    if season == "Holiday (Q4)":
        reasons.append("Holiday multiplier boosts demand by 25%.")
    if inventory < 300:
        reasons.append("Low inventory — raising price preserves stock.")
    st.markdown(insight(" · ".join(reasons), label="Reasoning"), unsafe_allow_html=True)

    # Scenario comparison
    st.markdown(section("Scenario Comparison", "Demand & revenue across the price range"), unsafe_allow_html=True)
    c7, c8 = st.columns(2)
    with c7:
        st.plotly_chart(demand_vs_price(a, b), use_container_width=True)
    with c8:
        labels = ["LOWER", "HOLD", "RAISE"]
        st.plotly_chart(bar(labels, q_row.tolist(),
                            f"Q-values for State #{state}", color="#8b5cf6"),
                        use_container_width=True)
