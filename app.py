"""PriceIQ — AI Pricing Intelligence (premium SaaS dashboard).

Architecture:
  • app.py                — entry, theme injection, top-nav, page dispatch
  • assets/styles.css     — full design system (glass, gradients, animations)
  • utils/loaders.py      — cached IO for npy / json / csv artifacts
  • utils/helpers.py      — UI primitives (hero, KPI cards, sparklines, insights)
  • utils/charts.py       — premium Plotly theme + chart factory
  • app_pages/*           — one render() per page
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import streamlit as st

from app_pages import (dashboard, data_studio, simulator, forecasting,
                       analytics, ai_insights, evaluation, training)

st.set_page_config(
    page_title="PriceIQ — AI Pricing Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject design system
_CSS = (Path(__file__).parent / "assets" / "styles.css").read_text()
st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

PAGES = {
    "Dashboard":    ("speedometer2", dashboard),
    "Data Studio":  ("cloud-upload", data_studio),
    "Forecasting":  ("graph-up",     forecasting),
    "AI Insights":  ("stars",        ai_insights),
    "Simulator":    ("sliders",      simulator),
    "Analytics":    ("diagram-3",    analytics),
    "Evaluation":   ("shield-check", evaluation),
    "Training":     ("cpu",          training),
}

PAGE_ICONS = {
    "Dashboard":   "⬡",
    "Data Studio": "↑",
    "Forecasting": "↗",
    "AI Insights": "✦",
    "Simulator":   "⊟",
    "Analytics":   "⊕",
    "Evaluation":  "◈",
    "Training":    "⬡",
}

# ─── Sidebar Navigation ─────────────────────────────────────────────
if "active_page" not in st.session_state:
    st.session_state.active_page = "Dashboard"
if "previous_page" not in st.session_state:
    st.session_state.previous_page = "Dashboard"
if "page_transition" not in st.session_state:
    st.session_state.page_transition = False

# Sidebar header
st.sidebar.markdown(
    """
    <div class="sb-brand">
      <div class="sb-logo"></div>
      <div>
        <div class="sb-title">PriceIQ</div>
        <div class="sb-sub">AI Pricing Intelligence</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("")  # spacing

# Navigation buttons with active state indicator
page_names = list(PAGES.keys())
for name in page_names:
    icon = PAGE_ICONS.get(name, "•")
    is_active = st.session_state.active_page == name
    
    if st.sidebar.button(
        f"{icon} {name}",
        key=f"nav_{name}",
        use_container_width=True,
    ):
        if st.session_state.active_page != name:
            st.session_state.previous_page = st.session_state.active_page
            st.session_state.page_transition = True
        st.session_state.active_page = name
        st.rerun()
    
    # Add active indicator line below button
    if is_active:
        st.sidebar.markdown(
            '<div style="margin: -8px 0 0 0; height: 2px; background: var(--accent); border-radius: 1px; animation: slideIn 300ms ease-out;"></div>',
            unsafe_allow_html=True
        )

st.sidebar.markdown("")  # spacing
st.sidebar.markdown(
    f'<div class="sb-foot" style="text-align: center;"><span class="sb-dot"></span> <strong>Live</strong> · {datetime.now().strftime("%H:%M")}</div>',
    unsafe_allow_html=True
)

# ─── Page Transition Container ──────────────────────────────────────
# Wrap page content for smooth transitions
page_key = st.session_state.active_page
st.markdown(f'<div class="page-transition" data-page="{page_key}">', unsafe_allow_html=True)

# ─── Dispatch ───────────────────────────────────────────────────────
choice = st.session_state.active_page
PAGES[choice][1].render()

st.markdown('</div>', unsafe_allow_html=True)
st.session_state.page_transition = False
