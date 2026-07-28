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
                        width='stretch')
        best = np.argmax(q, axis=1)
        raise_states = [i for i, a in enumerate(best) if a == 2]
        lower_states = [i for i, a in enumerate(best) if a == 0]
        hold_states  = [i for i, a in enumerate(best) if a == 1]
        dynamic_msg = (
            f"<strong>RAISE</strong> preferred in {len(raise_states)} state(s) "
            f"(states {raise_states}) — low-to-mid prices, margin capture. "
            f"<strong>LOWER</strong> preferred in {len(lower_states)} state(s) "
            f"(states {lower_states}) — high prices, demand recovery. "
            f"<strong>HOLD</strong> in {len(hold_states)} state(s)."
        )
        st.markdown(insight(dynamic_msg, label="Heatmap Reading"), unsafe_allow_html=True)

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
            st.dataframe(df, width='stretch', hide_index=True, height=380)
        with c2:
            counts = pd.Series([ACTION_LABELS[a] for a in best]).value_counts().to_dict()
            st.plotly_chart(action_distribution(counts), width='stretch')

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
                            color="#22d3ee"), width='stretch')
        st.plotly_chart(bar(actions, q.mean(axis=0).tolist(),
                            "Mean Q across all states", color="#f472b6"),
                        width='stretch')

    with tab4:
        path = Path("models/coefficients_by_product.json")
        if not path.exists():
            st.info(
                "Per-product elasticity data not found. "
                "To generate it, run this command in your terminal from the project folder:\n\n"
                "```\npython data_processing.py --input data/2025_retail_sample.csv "
                "--price-col price --quantity-col quantity --by product\n```"
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
                    st.dataframe(edf, width='stretch', hide_index=True, height=420)
                with c2:
                    top = edf.head(15)
                    st.plotly_chart(
                        bar(top["Product"].tolist(),
                            top["Elasticity (|b|)"].tolist(),
                            "Top-15 most price-elastic products",
                            color="#cc785c"),
                        width='stretch',
                    )
                st.markdown(insight(
                    f"Elasticity ranges from <strong>{edf['Elasticity (|b|)'].min():.2f}</strong> "
                    f"to <strong>{edf['Elasticity (|b|)'].max():.2f}</strong> across "
                    f"{len(edf)} products. Higher values mean demand drops more sharply "
                    "as price rises — those SKUs benefit most from price cuts, "
                    "while inelastic SKUs (low |b|) can absorb price increases.",
                    label="Reading"), unsafe_allow_html=True)



