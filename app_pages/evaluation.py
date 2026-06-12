"""Model evaluation — enterprise QA scorecards."""
from __future__ import annotations
import streamlit as st

from utils.loaders import load_evaluation_report
from utils.helpers import hero, kpi_card, status_pill, section, insight
from utils.charts import reward_distribution, bar


def render() -> None:
    st.markdown(hero(
        "Model Evaluation",
        "Convergence, stability, and reward statistics from the held-out evaluation "
        "run. System-health view for production readiness.",
        badge="QA · BENCHMARK",
    ), unsafe_allow_html=True)

    rep = load_evaluation_report()
    if not rep:
        st.error("evaluation_report.json not found in /reports.")
        return

    e = rep.get("evaluation", {})
    base = rep.get("baseline", {})
    issues = rep.get("pathologies", {}).get("issues", [])
    mean_r = e.get("mean_reward", 0.0); std_r = e.get("std_reward", 0.0)
    zero_r = e.get("zero_reward_count", 0); mean_p = e.get("mean_price", 0.0)
    mean_d = e.get("mean_demand", 0.0)
    fixed = base.get("fixed_optimal_expected_revenue", 0.0)
    converged = std_r < 0.5 and mean_r > 0
    lift = ((mean_r - fixed) / fixed * 100) if fixed else 0.0

    # System health strip
    health = [
        ("Convergence", "ok" if converged else "warn", "Stable" if converged else "Borderline"),
        ("Reward signal", "ok" if mean_r > 0 else "err", f"μ = {mean_r:.3f}"),
        ("Action coverage", "warn" if issues else "ok", "All actions used" if not issues else "Imbalance"),
        ("Variance", "ok" if std_r < 0.3 else "warn", f"σ = {std_r:.3f}"),
    ]
    pills = " ".join(status_pill(f"{n} · {v}", k) for n, k, v in health)
    st.markdown(f'<div class="pi-card"><h3>System Health</h3>{pills}</div>', unsafe_allow_html=True)

    st.markdown(section("Scorecards"), unsafe_allow_html=True)
    cards = [
        ("Mean reward", f"{mean_r:.4f}", None, False, "✦"),
        ("Std reward",  f"{std_r:.4f}",  "stability", False, "≈"),
        ("Mean price",  f"${mean_p:.2f}", None, False, "$"),
        ("Mean demand", f"{mean_d:.3f}", None, False, "≡"),
        ("Zero-reward events", f"{zero_r:,}", "edge cases", zero_r > 100, "⌀"),
        ("Lift vs baseline", f"{lift:+.1f}%", "vs fixed-optimal", lift < 0, "↑"),
    ]
    cols = st.columns(3)
    for i, (l, v, d, neg, ic) in enumerate(cards):
        with cols[i % 3]:
            st.markdown(kpi_card(l, v, d, neg, icon=ic), unsafe_allow_html=True)

    st.markdown(section("Distributions & Benchmarks"), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(reward_distribution(mean_r, std_r), use_container_width=True)
    with c2:
        st.plotly_chart(bar(["Learned policy", "Fixed-optimal"], [mean_r, fixed],
                            "Learned vs Fixed-Optimal Reward", color="#8b5cf6"),
                        use_container_width=True)

    if converged:
        st.markdown(insight(
            f"Policy <strong>converged</strong>: σ = {std_r:.3f} (&lt; 0.5) with positive "
            f"mean reward {mean_r:.3f}. Stable for production deployment.",
            label="System Verdict"), unsafe_allow_html=True)
    else:
        st.markdown(insight(
            f"Policy stability is <strong>borderline</strong>: σ = {std_r:.3f}. "
            "Recommend additional episodes or a slower epsilon decay before promotion.",
            label="System Verdict"), unsafe_allow_html=True)

    if issues:
        st.markdown(section("Pathologies Detected"), unsafe_allow_html=True)
        for i in issues:
            st.markdown(status_pill(i, "warn"), unsafe_allow_html=True)
