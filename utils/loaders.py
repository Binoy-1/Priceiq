"""Cached file loaders — keep IO out of page code."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
DATA = ROOT / "data"


@st.cache_data(show_spinner=False)
def load_q_table(path: str | None = None) -> np.ndarray | None:
    p = Path(path) if path else MODELS / "q_table.npy"
    if not p.exists():
        return None
    return np.load(p)


@st.cache_data(show_spinner=False)
def load_coefficients() -> np.ndarray | None:
    p = MODELS / "coefficients.npy"
    if not p.exists():
        return None
    return np.load(p)


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict | None:
    p = REPORTS / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_pricing_report() -> dict | None:
    return load_json("pricing_report.json")


@st.cache_data(show_spinner=False)
def load_evaluation_report() -> dict | None:
    return load_json("evaluation_report.json")


@st.cache_data(show_spinner=False)
def load_recommendations_df() -> pd.DataFrame:
    rep = load_pricing_report() or {}
    recs = rep.get("recommendations", [])
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"])
        df["quarter"] = df["date"].dt.quarter.map(lambda q: f"Q{q}")
        df["month"] = df["date"].dt.month_name()
    return df


@st.cache_data(show_spinner=False)
def load_retail_sample() -> pd.DataFrame:
    p = DATA / "2025_retail_sample.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()
