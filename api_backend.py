# FastAPI backend for the AQI dashboard.
# uvicorn api_backend:app --reload --port 8000

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import eda_utils
import model_service
import shap_utils
from alerts import build_alerts, classify_aqi
from dashboard_config import CITY, FORECAST_HORIZONS, TARGET_COL
import traceback

app = FastAPI(title="AQI Forecasting API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "city": CITY}

@app.get("/predict")
def predict(force_refresh: bool = False):
    """Real-time 24h/48h/72h AQI predictions from the latest Feature Store
    row, using the models registered in the Hopsworks Model Registry."""
    try:
        result = model_service.predict_next_3_days(force_refresh_models=force_refresh)
    except Exception as e:
        traceback.print_exc()  # prints full traceback to the Streamlit Cloud logs
        raise HTTPException(status_code=500, detail=str(e))

    flat_predictions = {k: v["predicted_aqi"] for k, v in result["predictions"].items()}
    for v in result["predictions"].values():
        v["category"] = classify_aqi(v["predicted_aqi"])

    result["current_category"] = classify_aqi(result["current_aqi"]) if result["current_aqi"] is not None else None
    result["alerts"] = build_alerts(flat_predictions)
    return result

@app.get("/eda/summary")
def eda_summary():
    df = eda_utils.load_eda_dataframe()
    return eda_utils.summary_stats(df)

@app.get("/eda/trend")
def eda_trend():
    df = eda_utils.load_eda_dataframe()
    daily = df.set_index("datetime")[TARGET_COL].resample("D").mean().reset_index()
    daily["datetime"] = daily["datetime"].astype(str)
    return daily.to_dict(orient="records")

@app.get("/eda/hourly")
def eda_hourly():
    df = eda_utils.load_eda_dataframe()
    df = df.copy()
    df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
    hourly = df.groupby("hour")[TARGET_COL].mean().reset_index()
    return hourly.to_dict(orient="records")

@app.get("/eda/monthly")
def eda_monthly():
    df = eda_utils.load_eda_dataframe()
    df = df.copy()
    df["month"] = pd.to_datetime(df["datetime"]).dt.month
    monthly = df.groupby("month")[TARGET_COL].agg(["mean", "min", "max"]).reset_index()
    return monthly.to_dict(orient="records")

@app.get("/shap/{horizon}")
def shap_explanation(horizon: int):
    if horizon not in FORECAST_HORIZONS:
        raise HTTPException(status_code=400, detail=f"horizon must be one of {FORECAST_HORIZONS}")
    try:
        _, fs, mr = model_service.get_hopsworks()
        bundle = model_service.load_model_for_horizon(mr, horizon)
        latest_row = model_service.get_latest_feature_row(fs)

        df = model_service.get_recent_feature_rows(fs)
        feature_cols = bundle["feature_columns"]
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=500,
                detail=f"Feature Store is missing columns the {horizon}h model expects: {missing_cols}. "
                       f"The model may have been trained on an older/different feature set than what's "
                       f"currently in the Feature Store."
            )
        background_df = df.dropna(subset=feature_cols).tail(300)
        if background_df.empty:
            raise HTTPException(status_code=500, detail="Not enough historical data for a SHAP background sample.")

        explanation = shap_utils.explain_prediction(
            bundle["model"], bundle["scaler"], feature_cols, background_df, latest_row,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    explanation["algorithm"] = bundle["algorithm"]
    explanation["horizon"] = horizon
    return explanation


@app.get("/models/{horizon}")
def models_for_horizon(horizon: int):
    if horizon not in FORECAST_HORIZONS:
        raise HTTPException(status_code=400, detail=f"horizon must be one of {FORECAST_HORIZONS}")
    _, fs, mr = model_service.get_hopsworks()
    return model_service.list_available_models(mr, horizon)
