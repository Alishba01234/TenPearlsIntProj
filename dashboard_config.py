"""
dashboard_config.py
Shared configuration for the AQI Forecasting Dashboard (Streamlit + FastAPI).

This file does NOT modify aqiPipeline.py / train_models.py -- it only holds
constants that need to match them (city key, horizons, target column) so the
dashboard stays consistent with how the Feature Store / Model Registry were
populated.
"""

# --- Must match aqiPipeline.py / train_models.py ---
CITY = "karachi"                     # matches city_key = args.city.lower().replace(" ", "_")
FORECAST_HORIZONS = (24, 48, 72)
TARGET_COL = "us_aqi"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_VERSION = 1

# --- Dashboard-only settings ---
LOCAL_FEATURES_CSV = "aqi_features.csv"   # already produced by aqiPipeline.py, read-only
MODEL_CACHE_DIR = "registry_cache"        # NEW folder for downloaded registry models --
                                           # separate from your existing model_aqi_karachi_*h folders

API_HOST = "127.0.0.1"
API_PORT = 8000
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"

# --- US EPA AQI breakpoints (matches the us_aqi scale from Open-Meteo) ---
AQI_BREAKPOINTS = [
    (0, 50, "Good", "#00e400", "Air quality is satisfactory, poses little or no risk."),
    (51, 100, "Moderate", "#ffff00", "Acceptable; may pose a risk for a very small group of people."),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00", "Sensitive groups may experience health effects."),
    (151, 200, "Unhealthy", "#ff0000", "Everyone may begin to experience health effects."),
    (201, 300, "Very Unhealthy", "#8f3f97", "Health alert: everyone may experience more serious effects."),
    (301, 500, "Hazardous", "#7e0023", "Health warning of emergency conditions -- entire population affected."),
]

ALERT_THRESHOLD = 150         # AQI at/above this triggers a "warning" alert
SEVERE_ALERT_THRESHOLD = 200  # AQI at/above this triggers a "severe" alert
