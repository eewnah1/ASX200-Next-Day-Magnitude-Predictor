"""Streamlit dashboard for the ASX200 Next-Day Magnitude Predictor."""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from asx200_mag_predictor.config import get_settings
from asx200_mag_predictor.data.fetchers import DataFetcher
from asx200_mag_predictor.logging_config import get_logger
from asx200_mag_predictor.scoring.engine import ScoringEngine
from asx200_mag_predictor.scoring.features import build_features
from asx200_mag_predictor.storage.models import init_db
from asx200_mag_predictor.storage.repository import Repository
from asx200_mag_predictor.timezone import to_sydney

logger = get_logger(__name__)

API_URL = os.environ.get("API_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="ASX200 Magnitude Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _api_get(path: str, timeout: int = 10):
    try:
        return requests.get(f"{API_URL}{path}", timeout=timeout).json()
    except Exception as exc:  # noqa: BLE001
        logger.debug("API GET %s failed: %s", path, exc)
        return None


def _api_post(path: str, timeout: int = 30):
    try:
        return requests.post(f"{API_URL}{path}", timeout=timeout).json()
    except Exception as exc:  # noqa: BLE001
        logger.debug("API POST %s failed: %s", path, exc)
        return None


def _local_predict(notes: str = "") -> dict:
    """Fallback in-process prediction if the API is unreachable."""
    init_db()
    raw = DataFetcher().fetch_all()
    features, flags = build_features(raw)
    prediction = ScoringEngine().predict(features, flags)
    if notes:
        prediction.notes.append(notes)
    prediction_id = Repository().save_prediction(prediction)
    return {"prediction_id": prediction_id, "prediction": prediction.model_dump()}


def _fmt_aest(iso: str) -> str:
    try:
        dt = to_sydney(datetime.fromisoformat(iso))
        return dt.strftime("%a %d %b %H:%M %Z")
    except Exception:  # noqa: BLE001
        return iso


# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("ASX200 Magnitude Predictor")
    st.caption(f"API: {API_URL}")
    st.caption(f"Internal TZ: {get_settings().tz}")
    st.divider()
    if st.button("Run prediction now", use_container_width=True):
        with st.spinner("Fetching data and scoring..."):
            result = _api_post("/predict")
            if not result:
                result = _local_predict()
            st.session_state["last_result"] = result
            st.rerun()
    notes = st.text_input("Optional note for manual run")


# ---------------- Main dashboard ----------------
st.header("Next trading day absolute move probability")

latest = _api_get("/predictions/latest")
if not latest:
    st.info("No prediction available. Run one from the sidebar.")
else:
    probs = latest["probabilities"]
    bucket = latest["bucket"]
    confidence = latest["confidence"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Low (<0.3%)", f"{probs['low']:.1%}")
    c2.metric("Mid (0.3-0.5%)", f"{probs['mid']:.1%}")
    c3.metric("High (>0.5%)", f"{probs['high']:.1%}")
    c4.metric("Confidence", f"{confidence:.0%}")

    gen = _fmt_aest(latest["generated_at"])
    st.markdown(f"**Primary bucket:** `{bucket}`  |  Generated: {gen}")

    # Factor contribution chart
    st.subheader("Factor contribution to large-move probability")
    fb = latest.get("factor_breakdown", {})
    if fb:
        fig = go.Figure(
            go.Bar(
                x=list(fb.keys()),
                y=list(fb.values()),
                marker_color="steelblue",
            )
        )
        fig.update_layout(
            yaxis_title="High-bucket logit contribution",
            xaxis_title="Factor",
            height=350,
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Data health
    st.subheader("Data source health")
    flags = latest.get("data_quality_flags", {})
    flag_df = pd.DataFrame(
        [{"Source": k, "Status": v} for k, v in flags.items()]
    )
    st.dataframe(flag_df, use_container_width=True, hide_index=True)

    # Notes
    with st.expander("Prediction notes"):
        for note in latest.get("notes", []):
            st.write(f"- {note}")

st.divider()

# ---------------- Historical predictions ----------------
st.subheader("Historical predictions vs actuals")
history = _api_get("/predictions?limit=50") or []
if history:
    hist_df = pd.json_normalize(history)
    if not hist_df.empty:
        hist_df["generated_at_aest"] = hist_df["generated_at"].apply(_fmt_aest)
        prob_cols = ["probabilities.low", "probabilities.mid", "probabilities.high"]
        cols = ["generated_at_aest", "bucket", "confidence", *prob_cols]
        display_df = hist_df[cols] if all(c in hist_df.columns for c in prob_cols) else hist_df
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Hit rate
        cal = _api_get("/calibration") or {}
        if cal and cal.get("total"):
            st.metric(
                "Calibration hit rate",
                f"{cal['hit_rate']:.1%}",
                f"{cal['correct']}/{cal['total']}",
            )

st.divider()

# ---------------- Calendar ----------------
st.subheader("Upcoming high-impact calendar proxy")
cal = _api_get("/calendar") or {}
if cal:
    st.json(cal)
else:
    st.info("Calendar data not available. Configure NEWSAPI_API_KEY.")

st.caption("All times displayed in Australia/Sydney (AEST/AEDT).")
