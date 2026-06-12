"""Data Studio — upload, schema mapping, AI cleaning, quality scoring."""
from __future__ import annotations
import streamlit as st
import pandas as pd

from utils.helpers import hero, kpi_card, section, insight, status_pill
from utils.data_pipeline import (read_uploaded, detect_roles, clean_dataset,
                                 aggregate_timeseries, ROLE_PATTERNS)


def _kpi_row(rep) -> None:
    cols = st.columns(4)
    items = [
        ("Quality score", f"{rep.quality_score:.1f}/100",
            f"{rep.rows_out:,} rows", rep.quality_score < 60, "✓"),
        ("Rows retained", f"{rep.rows_out:,}",
            f"of {rep.rows_in:,}", False, "≡"),
        ("Cleaning ops",
            f"{rep.duplicates_removed + rep.nulls_filled + rep.outliers_capped:,}",
            "dupes + nulls + outliers", False, "✦"),
        ("Dates parsed", f"{rep.dates_parsed:,}",
            "ISO normalized", False, "◷"),
    ]
    for i, (l, v, d, neg, ic) in enumerate(items):
        with cols[i]:
            st.markdown(kpi_card(l, v, d, neg, icon=ic), unsafe_allow_html=True)


def render() -> None:
    st.markdown(hero(
        "Data Studio",
        "Upload any pricing dataset (CSV, XLSX, JSON). PriceIQ infers schema, "
        "cleans, scores quality, and prepares it for forecasting and RL pricing.",
        badge="DATA · INGESTION",
    ), unsafe_allow_html=True)

    st.markdown(section("1 · Upload", "CSV / XLSX / JSON · up to 200 MB"),
                unsafe_allow_html=True)
    file = st.file_uploader("Drop a file", type=["csv", "xlsx", "xls", "json"],
                            label_visibility="collapsed")

    if file is None and "active_dataset" not in st.session_state:
        st.info("No dataset loaded. Drop a file above to begin, or load the bundled retail sample.")
        if st.button("Load 2025 retail sample"):
            from utils.loaders import load_retail_sample
            df = load_retail_sample()
            if df.empty:
                st.error("Sample missing.")
                return
            st.session_state["raw_dataset"] = df
            st.session_state["dataset_name"] = "2025_retail_sample.csv"
            st.rerun()
        return

    if file is not None:
        try:
            df = read_uploaded(file)
            st.session_state["raw_dataset"] = df
            st.session_state["dataset_name"] = file.name
            st.session_state.pop("active_dataset", None)
            st.session_state.pop("active_roles", None)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return

    raw = st.session_state.get("raw_dataset")
    if raw is None:
        return

    st.markdown(section("2 · Preview", f"{st.session_state.get('dataset_name', '')} · "
                        f"{len(raw):,} rows × {raw.shape[1]} cols"),
                unsafe_allow_html=True)
    st.dataframe(raw.head(20), use_container_width=True, hide_index=True)

    # ── Schema mapping ────────────────────────────────────────────────
    st.markdown(section("3 · Column mapping", "Auto-detected · override if needed"),
                unsafe_allow_html=True)
    detected = detect_roles(raw)
    cols = st.columns(len(ROLE_PATTERNS))
    roles: dict[str, str | None] = {}
    options = ["— none —"] + list(raw.columns.astype(str))
    for i, role in enumerate(ROLE_PATTERNS.keys()):
        with cols[i]:
            default = detected.get(role)
            idx = options.index(default) if default in options else 0
            choice = st.selectbox(role.title(), options, index=idx, key=f"map_{role}")
            roles[role] = None if choice == "— none —" else choice

    # ── Cleaning controls ─────────────────────────────────────────────
    st.markdown(section("4 · AI cleaning", "Choose strategy and run pipeline"),
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        outlier = st.selectbox("Outliers", ["cap", "drop", "keep"], index=0)
    with c2:
        freq = st.selectbox("Aggregate to", ["D", "W", "MS", "QS"], index=1,
                            format_func=lambda x: {"D": "Daily", "W": "Weekly",
                                                   "MS": "Monthly", "QS": "Quarterly"}[x])
    with c3:
        st.write("")
        st.write("")
        run = st.button("⚡  Run cleaning pipeline", use_container_width=True, type="primary")

    if run or st.session_state.get("active_dataset") is not None:
        if run:
            cleaned, rep = clean_dataset(raw, roles, outlier_strategy=outlier)
            ts = aggregate_timeseries(cleaned, roles, freq=freq)
            st.session_state["active_dataset"] = cleaned
            st.session_state["active_roles"] = roles
            st.session_state["active_ts"] = ts
            st.session_state["cleaning_report"] = rep
            st.session_state["agg_freq"] = freq

        rep = st.session_state["cleaning_report"]
        cleaned = st.session_state["active_dataset"]
        ts = st.session_state["active_ts"]

        _kpi_row(rep)

        if rep.notes:
            st.markdown(insight(" · ".join(rep.notes), label="Cleaning notes"),
                        unsafe_allow_html=True)
        if rep.issues:
            st.warning("⚠  " + " · ".join(rep.issues))

        # Status pills
        pills = []
        for role, col in st.session_state["active_roles"].items():
            kind = "ok" if col else "warn"
            pills.append(status_pill(f"{role}: {col or '—'}", kind))
        st.markdown(" ".join(pills), unsafe_allow_html=True)

        st.markdown(section("5 · Cleaned preview",
                            f"{len(cleaned):,} rows · ready for forecasting"),
                    unsafe_allow_html=True)
        st.dataframe(cleaned.head(25), use_container_width=True, hide_index=True)

        if not ts.empty:
            st.markdown(section("6 · Time-series aggregate",
                                f"{len(ts):,} periods @ {st.session_state['agg_freq']}"),
                        unsafe_allow_html=True)
            st.line_chart(ts.set_index("date")[
                [c for c in ["price", "revenue"] if c in ts.columns]
            ], use_container_width=True)

        from utils.exports import df_to_csv_bytes, df_to_xlsx_bytes
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇  Cleaned CSV", df_to_csv_bytes(cleaned),
                               file_name="cleaned_dataset.csv", mime="text/csv",
                               use_container_width=True)
        with c2:
            st.download_button("⬇  Cleaning report (XLSX)",
                               df_to_xlsx_bytes({
                                   "cleaned": cleaned,
                                   "timeseries": ts,
                                   "report": pd.DataFrame([rep.to_dict()]),
                               }),
                               file_name="priceiq_cleaning_report.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
