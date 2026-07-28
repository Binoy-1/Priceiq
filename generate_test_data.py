"""Generate synthetic retail dataset for PriceIQ testing."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(42)

products = ["Widget A", "Widget B", "Gadget X", "Gadget Y", "Premium Z"]
start = "2024-01-01"
dates = pd.date_range(start, periods=365, freq="D")

rows = []
for date in dates:
    for product in products:
        # Base price per product
        base_price = {"Widget A": 15, "Widget B": 25, "Gadget X": 45, "Gadget Y": 70, "Premium Z": 120}[product]
        # Random price variation ±20%
        price = round(base_price * rng.uniform(0.8, 1.2), 2)

        # Demand: log-log model with noise + seasonality
        season = 1 + 0.15 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
        noise = rng.normal(0, 0.15)
        demand = max(1, int(np.exp(4.5 - 0.8 * np.log(price) + noise) * season))
        demand += int(rng.integers(-2, 3))  # jitter
        demand = max(1, demand)

        revenue = round(price * demand, 2)
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "product": product,
            "price": price,
            "quantity": demand,
            "revenue": revenue,
        })

df = pd.DataFrame(rows)
out = ROOT / "data" / "test_retail_data.csv"
df.to_csv(out, index=False)
print(f"Wrote {len(df):,} rows to {out}")
print(f"Columns: {list(df.columns)}")
print(f"Products: {df['product'].unique().tolist()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
