import socket
import subprocess
import sys
import time

def _ensure_backend(host="127.0.0.1", port=8000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex((host, port)) == 0:
            return  # backend already running
    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api_backend:app", "--host", host, "--port", str(port)]
    )
    for _ in range(30):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) == 0:
                return
        time.sleep(0.5)

_ensure_backend()

"""
streamlit_app.py
Interactive AQI forecasting dashboard.

Run with:
    streamlit run streamlit_app.py
Requires api_backend.py to be running first:
    uvicorn api_backend:app --reload --port 8000
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from dashboard_config import API_BASE_URL, AQI_BREAKPOINTS, CITY, FORECAST_HORIZONS

st.set_page_config(page_title=f"{CITY.title()} AQI Forecast", page_icon="🌫️", layout="wide")


def api_get(path: str, params: dict = None):
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.error(
            f"Could not reach the API backend at {API_BASE_URL}{path}. "
            f"Is `uvicorn api_backend:app --port 8000` running?\n\n{e}"
        )
        return None


st.title(f"🌫️ {CITY.title()} Air Quality Forecast")
st.caption("Serverless ML stack: Open-Meteo → Hopsworks Feature Store → Hopsworks Model Registry → this dashboard.")

tab_forecast, tab_eda, tab_shap, tab_alerts = st.tabs(
    ["🔮 Forecast", "📊 EDA & Trends", "🧠 Model Explainability", "🚨 Alerts"]
)

# ---------------------------------------------------------------- Forecast
with tab_forecast:
    refresh = st.button("🔄 Refresh predictions (re-download models)")
    data = api_get("/predict", params={"force_refresh": refresh})
    if data:
        st.subheader(f"Reference reading: {data['reference_datetime']}  |  Current AQI: {data['current_aqi']}")
        cols = st.columns(3)
        for col, (label, pred) in zip(cols, data["predictions"].items()):
            with col:
                cat = pred["category"]
                st.metric(label=f"{label} forecast ({pred['algorithm']})", value=pred["predicted_aqi"])
                st.markdown(
                    f"<div style='background-color:{cat['color']};padding:8px;border-radius:6px;"
                    f"text-align:center;color:black;font-weight:bold'>{cat['category']}</div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"Target time: {pred['target_datetime']}")

        points = [{"time": data["reference_datetime"], "aqi": data["current_aqi"], "label": "now"}]
        for label, pred in data["predictions"].items():
            points.append({"time": pred["target_datetime"], "aqi": pred["predicted_aqi"], "label": label})
        chart_df = pd.DataFrame(points)
        chart_df["time"] = pd.to_datetime(chart_df["time"])
        fig = px.line(chart_df, x="time", y="aqi", markers=True, text="label", title="AQI: Now → Next 3 Days")
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------------- EDA
with tab_eda:
    summary = api_get("/eda/summary")
    if summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{summary['rows']:,}")
        c2.metric("Mean AQI", summary["mean_aqi"])
        c3.metric("Max AQI", summary["max_aqi"])
        c4.metric("% Unhealthy+", f"{summary['pct_unhealthy_or_worse']}%")
        st.caption(f"Data range: {summary['date_range'][0]} → {summary['date_range'][1]}")

    trend = api_get("/eda/trend")
    if trend:
        df_trend = pd.DataFrame(trend)
        df_trend["datetime"] = pd.to_datetime(df_trend["datetime"])
        st.plotly_chart(px.line(df_trend, x="datetime", y="us_aqi", title="Daily Average AQI"),
                         width='stretch')

    col1, col2 = st.columns(2)
    hourly = api_get("/eda/hourly")
    if hourly:
        df_hourly = pd.DataFrame(hourly)
        col1.plotly_chart(
            px.line(df_hourly, x="hour", y="us_aqi", markers=True, title="Average AQI by Hour of Day"),
            width='stretch',
        )

    monthly = api_get("/eda/monthly")
    if monthly:
        df_monthly = pd.DataFrame(monthly)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_monthly["month"], y=df_monthly["mean"], name="mean"))
        fig.add_trace(go.Scatter(x=df_monthly["month"], y=df_monthly["max"], name="max"))
        fig.add_trace(go.Scatter(x=df_monthly["month"], y=df_monthly["min"], name="min"))
        fig.update_layout(title="Monthly AQI Pattern (Seasonality)")
        col2.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------------- SHAP
with tab_shap:
    horizon = st.selectbox("Forecast horizon", FORECAST_HORIZONS, format_func=lambda h: f"{h}h")
    if st.button("Explain this model's prediction"):
        with st.spinner("Computing SHAP values..."):
            explanation = api_get(f"/shap/{horizon}")
        if explanation:
            st.write(f"Model: **{explanation['algorithm']}**")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Why this specific forecast** (local SHAP)")
                df_local = pd.DataFrame(explanation["local"])
                st.plotly_chart(
                    px.bar(df_local, x="shap_value", y="feature", orientation="h", title="Local feature contribution"),
                    width='stretch',
                )
            with c2:
                st.markdown("**What the model relies on overall** (global SHAP)")
                df_global = pd.DataFrame(explanation["global"])
                st.plotly_chart(
                    px.bar(df_global, x="mean_abs_shap", y="feature", orientation="h", title="Global feature importance"),
                    width='stretch',
                )

# ---------------------------------------------------------------- Alerts
with tab_alerts:
    data = api_get("/predict")
    if data:
        if data["alerts"]:
            for a in data["alerts"]:
                if a["severity"] == "severe":
                    st.error(f"🔴 {a['message']}")
                else:
                    st.warning(f"🟠 {a['message']}")
        else:
            st.success("✅ No hazardous AQI levels forecast in the next 3 days.")

    st.markdown("#### AQI Category Reference")
    legend_df = pd.DataFrame(AQI_BREAKPOINTS, columns=["low", "high", "category", "color", "message"])
    for _, row in legend_df.iterrows():
        st.markdown(
            f"<div style='background-color:{row['color']};padding:6px;border-radius:4px;"
            f"margin-bottom:4px;color:black'><b>{row['low']}-{row['high']}: {row['category']}</b> "
            f"— {row['message']}</div>",
            unsafe_allow_html=True,
        )
