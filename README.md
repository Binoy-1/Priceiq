# PriceIQ — RL Dynamic Pricing

Streamlit dashboard for the Q-Learning dynamic-pricing project.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Layout

```
app.py              # entry, sidebar nav, theme, page router
app_pages/              # one file per page (dashboard, simulator, ...)
utils/
  loaders.py        # cached file loaders (q_table, reports, csv)
  charts.py         # reusable Plotly chart builders
  helpers.py        # state binning, action labels, KPI math
models/             # q_table.npy, coefficients.npy  (your trained artifacts)
reports/            # evaluation_report.json, pricing_report.json
data/               # 2025_retail_sample.csv
agent.py, environment.py  # your RL engine, untouched, used by training page
```

## Architecture

- **Modular pages** — `app.py` is a thin router. Each page is a pure
  `render()` function. Adding a new page = drop a file in `app_pages/` and
  register it in `PAGES`.
- **Cached loaders** — `@st.cache_data` on JSON/npy/csv reads, so
  navigation never re-reads disk.
- **Chart builders** — every Plotly figure is built in `utils/charts.py`
  with the shared dark theme, so styling stays consistent.
- **RL engine untouched** — `agent.py` and `environment.py` are your
  original files. The Training page imports them directly.
