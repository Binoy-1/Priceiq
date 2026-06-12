"""Policy & Q-Table analytics."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils.loaders import load_q_table
from utils.helpers import hero, ACTION_LABELS, section, insight
from utils.charts import heatmap, action_distribution, bar


def render() -> None:
    st.markdown(hero(
        "Policy & Q-Table Analytics",
        "Inspect what the agent has learned. Each state corresponds to a price bin; "
        "each column is a discrete pricing action.",
        badge="POLICY · EXPLAINABILITY",
    ), unsafe_allow_html=True)

    q = load_q_table()
    if q is None:
        st.error("q_table.npy not found in /models.")
        return

    states = [f"S{i}" for i in range(q.shape[0])]
    actions = [ACTION_LABELS[i] for i in range(q.shape[1])]

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Q-Table Heatmap", "Best Actions", "Diagnostics", "Per-product Elasticity"]
    )

    with tab1:
        st.plotly_chart(heatmap(q, actions, states, "Q-Value Surface"),
                        use_container_width=True)
        st.markdown(insight(
            "Brighter cells = higher expected reward. Cyan dominates the upper-right "
            "quadrant — the policy has discovered that <strong>RAISE</strong> in mid-low "
            "states yields the strongest return.",
            label="Heatmap Reading"), unsafe_allow_html=True)

    with tab2:
        c1, c2 = st.columns([1.1, 1])
        best = np.argmax(q, axis=1)
        df = pd.DataFrame({
            "State": states,
            "Best action": [ACTION_LABELS[a] for a in best],
            "Max Q": np.max(q, axis=1).round(4),
            "Spread": (np.max(q, axis=1) - np.min(q, axis=1)).round(4),
        })
        with c1:
            st.dataframe(df, use_container_width=True, hide_index=True, height=380)
        with c2:
            counts = pd.Series([ACTION_LABELS[a] for a in best]).value_counts().to_dict()
            st.plotly_chart(action_distribution(counts), use_container_width=True)

        low = np.where(best == 0)[0].tolist()
        hold = np.where(best == 1)[0].tolist()
        rais = np.where(best == 2)[0].tolist()
        st.markdown(section("Policy Interpretation"), unsafe_allow_html=True)
        st.markdown(f"""
- **LOWER** preferred in states `{low}` — high prices, demand response yields more.
- **HOLD**  preferred in states `{hold}` — Q-values are flat, agent maintains.
- **RAISE** preferred in states `{rais}` — low-price bins, margin capture.
        """)

    with tab3:
        spread = (np.max(q, axis=1) - np.min(q, axis=1))
        st.plotly_chart(bar(states, spread.tolist(),
                            "Q-Value Spread by State (decisiveness)",
                            color="#22d3ee"), use_container_width=True)
        st.plotly_chart(bar(actions, q.mean(axis=0).tolist(),
                            "Mean Q across all states", color="#f472b6"),
                        use_container_width=True)

    with tab4:
        path = Path("models/coefficients_by_product.json")
        if not path.exists():
            st.info(
                "Run `python data_processing.py --input <csv> --by product` "
                "(or `multi_product.py`) to populate per-product elasticities."
            )
        else:
            data = json.loads(path.read_text())
            if not data:
                st.warning("coefficients_by_product.json is empty.")
            else:
                rows = []
                for prod, d in data.items():
                    rows.append({
                        "Product": prod,
                        "Elasticity (|b|)": round(abs(float(d.get("b", 0.0))), 3),
                        "Intercept a": round(float(d.get("a", 0.0)), 3),
                        "R²": round(float(d.get("r_squared", 0.0)), 3),
                        "n": int(d.get("n", 0)),
                    })
                edf = pd.DataFrame(rows).sort_values("Elasticity (|b|)", ascending=False)
                c1, c2 = st.columns([1.3, 1])
                with c1:
                    st.dataframe(edf, use_container_width=True, hide_index=True, height=420)
                with c2:
                    top = edf.head(15)
                    st.plotly_chart(
                        bar(top["Product"].tolist(),
                            top["Elasticity (|b|)"].tolist(),
                            "Top-15 most price-elastic products",
                            color="#cc785c"),
                        use_container_width=True,
                    )
                st.markdown(insight(
                    f"Elasticity ranges from <strong>{edf['Elasticity (|b|)'].min():.2f}</strong> "
                    f"to <strong>{edf['Elasticity (|b|)'].max():.2f}</strong> across "
                    f"{len(edf)} products. Higher values mean demand drops more sharply "
                    "as price rises — those SKUs benefit most from price cuts, "
                    "while inelastic SKUs (low |b|) can absorb price increases.",
                    label="Reading"), unsafe_allow_html=True)
