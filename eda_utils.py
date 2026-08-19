"""
eda_utils.py
Exploratory data analysis over the engineered feature set. Reads
aqi_features.csv (already produced by aqiPipeline.py) read-only -- it is
never modified. Falls back to a live Feature Store read if the CSV isn't
present in the folder.
"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard_config import LOCAL_FEATURES_CSV, TARGET_COL, CITY, FEATURE_GROUP_VERSION

_df_cache = {}


def load_eda_dataframe(force_refresh: bool = False) -> pd.DataFrame:
    if not force_refresh and "df" in _df_cache:
        return _df_cache["df"]

    if os.path.exists(LOCAL_FEATURES_CSV):
        df = pd.read_csv(LOCAL_FEATURES_CSV, parse_dates=["datetime"])
    else:
        from model_service import get_hopsworks
        _, fs, _ = get_hopsworks()
        fg = fs.get_feature_group(f"aqi_features_{CITY}", version=FEATURE_GROUP_VERSION)
        df = fg.read()
        df["datetime"] = pd.to_datetime(df["datetime"])

    df = df.sort_values("datetime").reset_index(drop=True)
    _df_cache["df"] = df
    return df


def summary_stats(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "date_range": [str(df["datetime"].min()), str(df["datetime"].max())],
        "current_aqi": float(df[TARGET_COL].dropna().iloc[-1]) if df[TARGET_COL].notna().any() else None,
        "mean_aqi": round(float(df[TARGET_COL].mean()), 1),
        "max_aqi": round(float(df[TARGET_COL].max()), 1),
        "min_aqi": round(float(df[TARGET_COL].min()), 1),
        "pct_unhealthy_or_worse": round(float((df[TARGET_COL] > 150).mean() * 100), 1),
    }


def daily_trend_figure(df: pd.DataFrame):
    daily = df.set_index("datetime")[TARGET_COL].resample("D").mean().reset_index()
    return px.line(daily, x="datetime", y=TARGET_COL,
                    title="Daily Average AQI Over Time", labels={TARGET_COL: "AQI", "datetime": "Date"})


def monthly_seasonality_figure(df: pd.DataFrame):
    tmp = df.copy()
    tmp["month_name"] = pd.to_datetime(tmp["datetime"]).dt.month_name()
    order = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    return px.box(tmp, x="month_name", y=TARGET_COL, category_orders={"month_name": order},
                  title="AQI Distribution by Month (Seasonality)")


def hourly_pattern_figure(df: pd.DataFrame):
    tmp = df.copy()
    tmp["hour"] = pd.to_datetime(tmp["datetime"]).dt.hour
    hourly = tmp.groupby("hour")[TARGET_COL].mean().reset_index()
    return px.line(hourly, x="hour", y=TARGET_COL, markers=True, title="Average AQI by Hour of Day")


def correlation_heatmap_figure(df: pd.DataFrame, top_n: int = 15):
    numeric_df = df.select_dtypes(include=[np.number])
    corrs = numeric_df.corr()[TARGET_COL].abs().sort_values(ascending=False)
    top_cols = corrs.head(top_n).index.tolist()
    corr_matrix = numeric_df[top_cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
        colorscale="RdBu", zmid=0,
    ))
    fig.update_layout(title=f"Correlation Heatmap (Top {top_n} Features vs {TARGET_COL})")
    return fig


def pollutant_distribution_figure(df: pd.DataFrame):
    cols = [c for c in ["pm2_5", "pm10", "nitrogen_dioxide", "ozone",
                         "carbon_monoxide", "sulphur_dioxide"] if c in df.columns]
    fig = go.Figure()
    for c in cols:
        fig.add_trace(go.Box(y=df[c], name=c))
    fig.update_layout(title="Pollutant Distributions")
    return fig
