"""AI Insights — auto-generated business commentary on the active dataset."""
from __future__ import annotations
import os
import streamlit as st

from utils.helpers import hero, section, insight, status_pill
from utils.ai import generate_insights


def _no_data() -> None:
    st.markdown(hero(
        "AI Insights",
        "Upload a dataset in Data Studio to unlock automated business insights.",
        badge="AI · INSIGHTS",
    ), unsafe_allow_html=True)
    st.info("No active dataset. Visit the Data Studio page first.")


def render() -> None:
    df = st.session_state.get("active_dataset")
    roles = st.session_state.get("active_roles")
    ts = st.session_state.get("active_ts")
    if df is None or roles is None:
        return _no_data()

    st.markdown(hero(
        "AI Insights",
        "Strategic commentary generated from your cleaned dataset. "
        "Set OPENAI_API_KEY or GEMINI_API_KEY to upgrade from rule-based to LLM-powered.",
        badge="AI · STRATEGIST",
    ), unsafe_allow_html=True)

    # Provider status
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    pills = [
        status_pill(f"OpenAI {'connected' if has_openai else 'not set'}",
                    "ok" if has_openai else "warn"),
        status_pill(f"Gemini {'connected' if has_gemini else 'not set'}",
                    "ok" if has_gemini else "warn"),
        status_pill("Rule-based fallback", "ok"),
    ]
    st.markdown(" ".join(pills), unsafe_allow_html=True)

    if st.button("✦  Generate insights", type="primary"):
        with st.spinner("Analyzing dataset…"):
            insights, source = generate_insights(df, roles, ts if ts is not None else df)
            st.session_state["last_insights"] = insights
            st.session_state["insights_source"] = source

    insights = st.session_state.get("last_insights")
    source = st.session_state.get("insights_source")
    if not insights:
        st.caption("Click the button above to run analysis.")
        return

    st.markdown(section("Generated insights", f"Source: {source}"),
                unsafe_allow_html=True)
    for ins in insights:
        st.markdown(insight(ins.get("text", ""), label=ins.get("label", "Insight")),
                    unsafe_allow_html=True)

    # Export to PDF
    from utils.exports import report_to_pdf_bytes
    sections = [(ins.get("label", "Insight"), ins.get("text", "")) for ins in insights]
    pdf = report_to_pdf_bytes("AI Insights Report", sections,
                              tables={"Cleaned dataset (head)": df.head(40)})
    st.download_button("⬇  Download PDF report", pdf,
                       file_name="priceiq_ai_insights.pdf",
                       mime="application/pdf")
