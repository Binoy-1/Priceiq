"""AI insights — OpenAI/Gemini if available, rule-based fallback otherwise.

Reads `OPENAI_API_KEY` or `GEMINI_API_KEY` from env. No keys required for the
rule-based engine, which produces realistic insights from dataset statistics.
"""
from __future__ import annotations
import os
import json
from typing import Any
import numpy as np
import pandas as pd


def _summary_payload(df: pd.DataFrame, roles: dict[str, str | None],
                     ts: pd.DataFrame) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "roles": {k: v for k, v in roles.items() if v},
    }
    price = roles.get("price")
    qty = roles.get("quantity")
    rev = roles.get("revenue")
    date = roles.get("date")

    if price and price in df:
        s = pd.to_numeric(df[price], errors="coerce").dropna()
        payload["price"] = {
            "min": float(s.min()), "max": float(s.max()),
            "mean": float(s.mean()), "std": float(s.std()),
            "p25": float(s.quantile(0.25)), "p75": float(s.quantile(0.75)),
        }
    if qty and qty in df:
        s = pd.to_numeric(df[qty], errors="coerce").dropna()
        payload["quantity"] = {"sum": float(s.sum()), "mean": float(s.mean())}
    if rev and rev in df:
        s = pd.to_numeric(df[rev], errors="coerce").dropna()
        payload["revenue"] = {"total": float(s.sum()), "mean": float(s.mean())}
    if date and date in df and not ts.empty:
        payload["timeseries"] = {
            "first": str(ts["date"].iloc[0].date()),
            "last": str(ts["date"].iloc[-1].date()),
            "periods": int(len(ts)),
        }
        if "revenue" in ts and len(ts) >= 4:
            half = len(ts) // 2
            growth = (ts["revenue"].tail(half).sum() - ts["revenue"].head(half).sum()) / max(
                1.0, ts["revenue"].head(half).sum())
            payload["revenue_growth_2H_vs_1H_%"] = round(growth * 100, 2)
    return payload


def _rule_based(payload: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    p = payload.get("price")
    if p:
        spread = (p["p75"] - p["p25"]) / max(1e-9, p["mean"]) * 100
        if spread > 40:
            out.append({"label": "High price dispersion",
                        "text": f"Inter-quartile price spread is {spread:.0f}% of the mean. "
                                "Segmenting by product or customer cohort will likely improve forecast accuracy."})
        else:
            out.append({"label": "Stable pricing",
                        "text": f"Prices cluster tightly (IQR = {spread:.0f}% of mean) — a single global "
                                "policy is appropriate."})
    g = payload.get("revenue_growth_2H_vs_1H_%")
    if g is not None:
        if g > 5:
            out.append({"label": "Positive momentum",
                        "text": f"Revenue is up {g:.1f}% in the second half vs first half of the period. "
                                "Consider gradual price increases on top-elasticity SKUs."})
        elif g < -5:
            out.append({"label": "Revenue contraction",
                        "text": f"Revenue is down {g:.1f}% in the second half. Investigate seasonal "
                                "factors before adjusting price; demand-side actions may be more effective."})
        else:
            out.append({"label": "Flat trajectory",
                        "text": f"Revenue is roughly flat ({g:+.1f}%). The dataset is in steady-state — "
                                "good baseline for testing controlled price experiments."})
    if payload.get("timeseries", {}).get("periods", 0) >= 52:
        out.append({"label": "Strong seasonality candidate",
                    "text": "More than a year of data — Holt-Winters or Prophet with yearly seasonality "
                            "will outperform naive trend models."})
    if "quantity" not in payload:
        out.append({"label": "Limited demand signal",
                    "text": "No quantity column was detected — elasticity and revenue optimization will "
                            "fall back on price-only signals. Mapping a units/orders column unlocks RL pricing."})
    if not out:
        out.append({"label": "Insufficient signal",
                    "text": "Not enough data after cleaning to generate confident insights. Upload more "
                            "rows or map additional columns."})
    return out


def _try_openai(payload: dict[str, Any]) -> list[dict[str, str]] | None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        prompt = (
            "You are a senior pricing strategist. Given this dataset summary (JSON), produce "
            "exactly 4 concise, action-oriented business insights. Return strict JSON: "
            '{"insights":[{"label":"...","text":"..."}, ...]}.\n\n'
            f"DATA:\n{json.dumps(payload, default=str)}"
        )
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        data = json.loads(resp.choices[0].message.content)
        return data.get("insights")
    except Exception:
        return None


def _try_gemini(payload: dict[str, Any]) -> list[dict[str, str]] | None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel(os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"))
        prompt = (
            "You are a senior pricing strategist. Return STRICT JSON only with key 'insights' "
            "as an array of 4 objects {label, text}. No prose.\n\n"
            f"DATA:\n{json.dumps(payload, default=str)}"
        )
        out = model.generate_content(prompt).text.strip()
        if out.startswith("```"):
            out = out.strip("`").split("\n", 1)[1]
        data = json.loads(out)
        return data.get("insights")
    except Exception:
        return None


def generate_insights(df: pd.DataFrame, roles: dict[str, str | None],
                      ts: pd.DataFrame) -> tuple[list[dict[str, str]], str]:
    """Returns (insights, source) where source ∈ {openai, gemini, rules}."""
    payload = _summary_payload(df, roles, ts)
    for fn, src in ((_try_openai, "openai"), (_try_gemini, "gemini")):
        result = fn(payload)
        if result:
            return result, src
    return _rule_based(payload), "rules"
