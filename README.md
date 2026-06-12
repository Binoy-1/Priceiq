# 💹 PriceIQ — Dynamic Pricing using Reinforcement Learning

> A Q-Learning agent that learns optimal pricing for e-commerce products,
> deployed as a production-quality Streamlit dashboard.

## 🎯 Business Problem
Static pricing rules don't adapt to demand signals. This system trains an 
RL agent to discover the revenue-maximising price through trial and error —
the same approach used by Amazon, Uber, and major airlines.

## 📊 Results
- Optimal price discovered: $12.88
- Revenue lift vs baseline: +22.5%
- Training convergence: ~episode 200 of 1500
- Demand model R²: fitted from UCI Online Retail II dataset

## 🛠️ Tech Stack
Python · NumPy · Pandas · Scikit-learn · Gymnasium · Streamlit · Plotly

## 🚀 Quick Start
```bash
git clone https://github.com/YOUR-USERNAME/PriceIQ.git
cd PriceIQ
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure
PriceIQ/

├── agent.py              # Q-Learning agent (Bellman updates)

├── environment.py        # Market simulation (log-log demand)

├── train.py              # Training pipeline

├── test_policy.py        # Policy evaluation

├── data_processing.py    # UCI dataset → demand coefficients

├── predict_2025_pricing.py # 52-week recommendations

├── app.py                # Streamlit entry point

├── app_pages/            # 8 dashboard pages

├── utils/                # Charts, loaders, forecasting

└── models/               # Trained Q-table + coefficients
## 🧠 How It Works
| Component | Details |
|-----------|---------|
| State | 9 discrete price levels ($1–$20) |
| Actions | Lower / Hold / Raise price |
| Reward | price × demand (revenue) |
| Algorithm | Tabular Q-Learning |
| Demand model | log(Q) = a − b·log(P) fitted from real data |

## 📈 Dashboard Pages
1. **Dashboard** — KPIs, revenue trends, demand curve
2. **Simulator** — Input market conditions, get live recommendation
3. **Analytics** — Q-table heatmap, learned policy visualisation
4. **Evaluation** — Model diagnostics, convergence analysis
5. **Forecasting** — 52-week revenue forecast (Holt-Winters/ARIMA/Prophet)
6. **AI Insights** — Auto-generated business commentary
7. **Data Studio** — Upload new data, refit demand model
8. **Training** — Retrain agent live from the UI

## ⚠️ Dataset
The UCI Online Retail II dataset is not included (licensing).
Download from: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
Place at: data/online_retail_II.csv
Then run: python data_processing.py --input data/online_retail_II.csv
