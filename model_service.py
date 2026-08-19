"""
model_service.py
Loads the latest features (Hopsworks Feature Store) and trained models
(Hopsworks Model Registry), and computes real-time next-3-day AQI
predictions.

Read-only with respect to aqiPipeline.py / train_models.py: it imports the
KerasWrapper class from train_models.py instead of redefining it, so a
loaded neural-net model behaves identically to how it was trained. Neither
file is modified or re-run.
"""
import os
from datetime import timedelta

import joblib
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import hopsworks

from dashboard_config import (
    CITY, FORECAST_HORIZONS, TARGET_COL, FEATURE_GROUP_VERSION, MODEL_CACHE_DIR,
)

# Reused, unmodified, from your existing training script.
from train_models import KerasWrapper  # noqa: F401

_connection_cache = {}
_model_cache = {}


def get_hopsworks():
    """Logs into Hopsworks once per process and caches the connection."""
    if "project" in _connection_cache:
        return _connection_cache["project"], _connection_cache["fs"], _connection_cache["mr"]

    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("HOPSWORKS_API_KEY not found -- check your .env file")

    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    _connection_cache.update({"project": project, "fs": fs, "mr": mr})
    return project, fs, mr


def get_latest_feature_row(fs, city: str = CITY, lookback_hours: int = 72) -> pd.Series:
    """Reads the most recent row from the feature group. Tries a
    time-filtered read first (fast, avoids pulling the whole history);
    falls back to a full read if the filter isn't usable for some reason."""
    fg = fs.get_feature_group(f"aqi_features_{city}", version=FEATURE_GROUP_VERSION)
    df = None
    try:
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - timedelta(hours=lookback_hours + 24)
        df = fg.filter(fg.datetime >= cutoff.strftime("%Y-%m-%d %H:%M:%S")).read()
    except Exception:
        df = None
    if df is None or df.empty:
        df = fg.read()

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Feature group returned no rows -- has the feature pipeline run yet?")
    return df.iloc[-1]


def _load_local_model_dir(local_dir: str):
    """Loads model + scaler from a directory laid out the same way
    register_model() in train_models.py writes it
    (model.pkl/scaler.pkl, or keras_model.keras/scaler.pkl)."""
    scaler = joblib.load(os.path.join(local_dir, "scaler.pkl"))
    keras_path = os.path.join(local_dir, "keras_model.keras")
    if os.path.exists(keras_path):
        import tensorflow as tf
        model = KerasWrapper(tf.keras.models.load_model(keras_path))
        algorithm = "neural_network"
    else:
        model = joblib.load(os.path.join(local_dir, "model.pkl"))
        algorithm = type(model).__name__
    return model, scaler, algorithm


def load_model_for_horizon(mr, horizon: int, city: str = CITY, force_refresh: bool = False) -> dict:
    """Downloads (once, then caches) the registered model for a horizon
    from the Model Registry and loads it into memory. Downloaded artifacts
    land under MODEL_CACHE_DIR/, kept separate from the model_aqi_{city}_{h}h/
    folders train_models.py already wrote locally when it trained."""
    cache_key = f"{city}_{horizon}"
    if not force_refresh and cache_key in _model_cache:
        return _model_cache[cache_key]

    model_name = f"aqi_{city}_{horizon}h"

    models = mr.get_models(name=model_name)
    if not models:
        raise RuntimeError(
            f"No registered model found in the Model Registry for '{model_name}'. "
            f"Has train_models.py been run and registered models yet?"
        )
    best = max(models, key=lambda m: m.version)

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    local_dir = best.download()  # hsml handles its own local caching path

    model, scaler, algorithm = _load_local_model_dir(local_dir)
    result = {
        "model": model,
        "scaler": scaler,
        "algorithm": algorithm,
        "version": best.version,
        "feature_columns": list(scaler.feature_names_in_),
    }
    _model_cache[cache_key] = result
    return result


def get_recent_feature_rows(fs, city: str = CITY, n: int = 300, lookback_days: int = 45) -> pd.DataFrame:
    """Pulls a batch of recent rows straight from the Feature Store (not the
    local aqi_features.csv, which can be a stale snapshot missing columns
    that were added/changed since it was last exported) so background data
    for SHAP is guaranteed to match exactly what the models were trained on."""
    fg = fs.get_feature_group(f"aqi_features_{city}", version=FEATURE_GROUP_VERSION)
    df = None
    try:
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - timedelta(days=lookback_days)
        df = fg.filter(fg.datetime >= cutoff.strftime("%Y-%m-%d %H:%M:%S")).read()
    except Exception:
        df = None
    if df is None or df.empty:
        df = fg.read()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df.tail(n * 2)  # extra buffer, since some rows will drop out for missing lags


def list_available_models(mr, horizon: int, city: str = CITY) -> list:
    """All registered versions for this horizon. Currently there is one
    model name per horizon (the CV winner from train_models.py), so this
    mainly returns its version history over time -- but if train_models.py
    is later extended to register multiple algorithms per horizon under
    different names (e.g. aqi_karachi_24h_ridge, aqi_karachi_24h_xgboost),
    this and load_model_for_horizon() are the two places to widen the
    lookup, with no dashboard-layer changes needed elsewhere."""
    model_name = f"aqi_{city}_{horizon}h"
    try:
        models = mr.get_models(name=model_name)
    except Exception:
        return []
    return sorted(
        [{"name": model_name, "version": m.version, "metrics": m.training_metrics} for m in models],
        key=lambda d: d["version"], reverse=True,
    )


def predict_next_3_days(force_refresh_models: bool = False) -> dict:
    """Returns predictions for each of the 3 forecast horizons, computed
    from the latest feature row available in the Feature Store."""
    project, fs, mr = get_hopsworks()
    latest_row = get_latest_feature_row(fs)
    reference_time = pd.to_datetime(latest_row["datetime"])

    predictions = {}
    for horizon in FORECAST_HORIZONS:
        bundle = load_model_for_horizon(mr, horizon, force_refresh=force_refresh_models)
        feature_cols = bundle["feature_columns"]

        missing = [c for c in feature_cols if c not in latest_row.index]
        if missing:
            raise RuntimeError(
                f"Latest feature row is missing columns needed by the {horizon}h model: {missing}"
            )

        X = pd.DataFrame([latest_row[feature_cols].values], columns=feature_cols)
        X_scaled = bundle["scaler"].transform(X)
        pred_value = float(bundle["model"].predict(X_scaled)[0])

        predictions[f"{horizon}h"] = {
            "horizon_hours": horizon,
            "target_datetime": (reference_time + timedelta(hours=horizon)).isoformat(),
            "predicted_aqi": round(pred_value, 1),
            "algorithm": bundle["algorithm"],
        }

    current_aqi = latest_row.get(TARGET_COL)
    return {
        "city": CITY,
        "reference_datetime": reference_time.isoformat(),
        "current_aqi": float(current_aqi) if pd.notna(current_aqi) else None,
        "predictions": predictions,
    }