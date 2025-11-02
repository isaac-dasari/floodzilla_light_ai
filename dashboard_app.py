# dashboard_app.py
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import pyarrow.dataset as ds
import streamlit as st

DATA_DIR = Path("dataset_parquet")
FORECAST_DIR = Path("forecasts_parquet")
SUMMARY_DIR = Path("summaries")

st.set_page_config(page_title="Floodzilla (Light AI)", layout="wide")

st.title("🌊 Floodzilla – Community Dashboard (Low-Cost AI)")

@st.cache_data
def list_gauges():
    if not DATA_DIR.exists():
        return []
    gauges = set()
    for path in DATA_DIR.glob("gauge_id=*"):
        if path.is_dir():
            parts = path.name.split("=", 1)
            if len(parts) == 2 and parts[1]:
                gauges.add(parts[1])
    return sorted(gauges)

@st.cache_data
def load_series(gid: str, days:int=30):
    dataset = ds.dataset(
        str(DATA_DIR),
        format="parquet",
        partitioning="hive",
        exclude_invalid_files=True,
    )
    df = dataset.to_table(filter=(ds.field("gauge_id")==gid)).to_pandas()
    df["ds"] = pd.to_datetime(df["ds"], utc=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    return df[df["ds"]>=cutoff].sort_values("ds")

@st.cache_data
def load_forecast(gid: str):
    if not FORECAST_DIR.exists(): return pd.DataFrame()
    dataset = ds.dataset(
        str(FORECAST_DIR),
        format="parquet",
        partitioning="hive",
        exclude_invalid_files=True,
    )
    try:
        df = dataset.to_table(filter=(ds.field("gauge_id")==gid)).to_pandas()
        df["ds"] = pd.to_datetime(df["ds"], utc=True)
        return df.sort_values("ds")
    except Exception:
        return pd.DataFrame()

def simple_risk(df: pd.DataFrame) -> float:
    if df.empty: return 0.0
    x = df["waterHeight"]
    p95 = np.nanpercentile(x, 95) if x.notna().sum()>10 else x.max()
    level = x.iloc[-1] / (p95 if p95>0 else (x.iloc[-1] or 1))
    roc = x.diff().iloc[-8:].mean() if x.shape[0]>8 else 0.0   # last ~2h avg rise
    flags = (df["flag_jump"].iloc[-24:].sum() + df["flag_outlier"].iloc[-24:].sum()) if df.shape[0]>24 else 0
    score = 0.6*min(2.0, level) + 0.3*max(0.0, roc) + 0.1*min(1.0, flags/4.0)
    return float(np.clip(score, 0, 3))

left, right = st.columns([1,2])
gauges = list_gauges()
gid = left.selectbox("Gauge", gauges)
days = left.slider("Days to show", 3, 180, 30)

df = load_series(gid, days)
st.subheader(f"Gauge {gid} — last {days} days")
st.line_chart(df.set_index("ds")["waterHeight"])

risk = simple_risk(df)
left.metric("Risk score (0–3)", f"{risk:.2f}", help="Heuristic combining level vs P95, recent rise, and anomaly flags.")

# Forecast overlay
fc = load_forecast(gid)
if not fc.empty:
    st.subheader("Prophet forecast (next horizon)")
    st.line_chart(fc.set_index("ds")[["yhat","yhat_lower","yhat_upper"]])

# Summary
st.subheader("Weekly Summary (LLM)")
latest = sorted(SUMMARY_DIR.glob("weekly_*.txt"))[-1] if SUMMARY_DIR.exists() and list(SUMMARY_DIR.glob("weekly_*.txt")) else None
if latest:
    txt = latest.read_text()
    # filter lines for this gauge if present
    lines = [ln for ln in txt.splitlines() if gid in ln or ln.startswith("- ")==False]
    st.text("\n".join(lines[:8]))
else:
    st.info("No LLM summary file yet. Run `python summaries_strands.py`.")
