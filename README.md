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
