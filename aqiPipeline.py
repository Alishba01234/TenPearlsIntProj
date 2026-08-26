"""
End-to-end AQI forecasting pipeline -- fetch, clean, engineer, store, split.
Combines what used to be two scripts (aqi_pipeline.py + create_train_test_split.py)
into one continuous run:
  1. Fetch raw weather + air quality data (Open-Meteo), auto date range =
     today going back N years (default 4). No dates typed in manually.
  2. Clean the raw data (duplicates, gaps, interpolation, negative-value clipping).
  3. Engineer features: time-based (hour, day, month) + derived (AQI lag
     features at each forecast horizon, rolling mean, AQI change rate) +
     per-horizon future-weather features ({var}_future_{h}h) so each of
     the 3 forecast horizons has something to distinguish it from the
     others -- see add_future_weather_features() for an important caveat
     about how these are approximated during training vs. at inference.
     Periodic columns (hour, month, wind direction) are sin/cos-encoded
     via add_cyclical_features() so wraparound (23h->0h, Dec->Jan,
     360deg->0deg) is represented correctly instead of as a large jump.
  4. Save ONE combined CSV (raw + engineered columns together).
  5. Upload to the Hopsworks Feature Store (deletes + recreates the
     feature group each run, so version always stays 1).
  6. Verify the upload by polling until materialization finishes and
     reading the data back.
  7. Create a chronological 70/15/15 train/validation/test split -- both
     as local CSVs and as a registered Hopsworks Feature View split.
Usage:
    python aqi_pipeline.py --city "Karachi"
    python aqi_pipeline.py --lat 24.8608 --lon 67.0104 --skip-hopsworks
"""

import argparse
import logging
import os
import time
from datetime import datetime, timedelta
import hopsworks
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logging.getLogger("hsfs").setLevel(logging.CRITICAL)
logging.getLogger("hopsworks").setLevel(logging.CRITICAL)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

AIR_QUALITY_MIN_DATE = "2022-08-01"  # CAMS global data doesn't exist before this
CHUNK_DAYS = 90

WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "pressure_msl", "cloud_cover", "wind_speed_10m", "wind_direction_10m",
    "dew_point_2m",           # narrow dew-point spread -> fog/inversion risk
    "apparent_temperature",   # combines temp+humidity+wind nonlinearly
    "vapour_pressure_deficit",  # atmospheric dryness
    "wind_gusts_10m",         # turbulence/mixing beyond mean wind speed
    "cloud_cover_low",        # low cloud specifically relates to boundary-layer fog/inversions
    "cloud_cover_mid",
    "cloud_cover_high",
    "surface_pressure",       # distinct from pressure_msl (elevation-adjusted)
]
AIR_QUALITY_VARS = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "us_aqi",
]

TARGET_COL = "us_aqi"
LAG_HOURS = [24, 48, 72]  # matches the 3 forecast horizons exactly
EXTRA_LAG_HOURS = [6, 12, 36, 60]  # finer-grained trend signal between horizons
ROLLING_WINDOWS = [24]
LONG_ROLLING_WINDOW_HOURS = 168  # 7 days -- longer-horizon trend signal for the 48h/72h models
FORECAST_HORIZONS = (24, 48, 72)
TARGET_COLS = [f"target_{TARGET_COL}_{h}h_ahead" for h in FORECAST_HORIZONS]
UPWIND_BEARINGS = {"N": 0, "E": 90, "S": 180, "W": 270}
UPWIND_DISTANCE_KM = 50

EDGE_SENSITIVE_COLS = [
    "aqi_lag_72h", "aqi_rolling_mean_24h",
    f"aqi_rolling_mean_{LONG_ROLLING_WINDOW_HOURS}h",
    f"pm2_5_rolling_mean_{LONG_ROLLING_WINDOW_HOURS}h",
    f"pm10_rolling_mean_{LONG_ROLLING_WINDOW_HOURS}h",
] + TARGET_COLS

# FETCHING
def geocode_city(city_name: str) -> tuple[float, float]:
    resp = requests.get(GEOCODING_URL, params={"name": city_name, "count": 1})
    resp.raise_for_status()
    results = resp.json().get("results")
    if not results:
        raise ValueError(f"Could not geocode city: {city_name}")
    r = results[0]
    print(f"Resolved '{city_name}' -> {r['name']}, {r.get('country')} "
          f"({r['latitude']}, {r['longitude']})")
    return r["latitude"], r["longitude"]

def date_chunks(start: str, end: str, chunk_days: int = CHUNK_DAYS):
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    cur = start_dt
    while cur <= end_dt:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end_dt)
        yield cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cur = chunk_end + timedelta(days=1)

def fetch_chunk(url: str, params: dict, retries: int = 3) -> dict:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"  attempt {attempt} failed: {e}")
            if attempt == retries:
                raise
            time.sleep(2 * attempt)

def fetch_weather(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    frames = []
    for c_start, c_end in date_chunks(start, end):
        print(f"Weather: {c_start} -> {c_end}")
        data = fetch_chunk(WEATHER_URL, {
            "latitude": lat, "longitude": lon,
            "start_date": c_start, "end_date": c_end,
            "hourly": ",".join(WEATHER_VARS),
            "timezone": "auto",
        })
        frames.append(pd.DataFrame(data["hourly"]))
    return pd.concat(frames, ignore_index=True)

def fetch_air_quality(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    if start < AIR_QUALITY_MIN_DATE:
        print(f"Note: air quality data only available from {AIR_QUALITY_MIN_DATE}, "
              f"adjusting start date (was {start})")
        start = AIR_QUALITY_MIN_DATE

    frames = []
    for c_start, c_end in date_chunks(start, end):
        print(f"Air quality: {c_start} -> {c_end}")
        data = fetch_chunk(AIR_QUALITY_URL, {
            "latitude": lat, "longitude": lon,
            "start_date": c_start, "end_date": c_end,
            "hourly": ",".join(AIR_QUALITY_VARS),
            "domain": "cams_global",
            "timezone": "auto",
        })
        frames.append(pd.DataFrame(data["hourly"]))
    return pd.concat(frames, ignore_index=True)

def offset_coordinates(lat: float, lon: float, bearing_deg: float, distance_km: float) -> tuple[float, float]:
    """Destination point at a given compass bearing and great-circle
    distance from (lat, lon). Standard spherical-earth formula."""
    R = 6371.0
    lat1, lon1, brng = map(np.radians, (lat, lon, bearing_deg))
    d_r = distance_km / R
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d_r) + np.cos(lat1) * np.sin(d_r) * np.cos(brng))
    lon2 = lon1 + np.arctan2(
        np.sin(brng) * np.sin(d_r) * np.cos(lat1),
        np.cos(d_r) - np.sin(lat1) * np.sin(lat2),
    )
    return float(np.degrees(lat2)), float(np.degrees(lon2))

def fetch_upwind_air_quality(lat: float, lon: float, start: str, end: str,
                              bearings: dict = UPWIND_BEARINGS,
                              distance_km: float = UPWIND_DISTANCE_KM) -> pd.DataFrame:
    """Fetches pm2_5 + us_aqi at fixed points around the city (one fetch per
    bearing, same chunking/retry path as the main fetch_air_quality). Costs
    len(bearings) extra full air-quality fetches -- with the default 4
    bearings this roughly ~5x's total air-quality fetch time, so it's kept
    as its own function that can be skipped via --skip-upwind."""
    merged = None
    for name, bearing in bearings.items():
        plat, plon = offset_coordinates(lat, lon, bearing, distance_km)
        print(f"Upwind point '{name}' (bearing {bearing} deg, {distance_km}km) -> ({plat:.4f}, {plon:.4f})")
        dir_df = fetch_air_quality(plat, plon, start, end)[["time", "pm2_5", "us_aqi"]]
        dir_df = dir_df.rename(columns={"pm2_5": f"pm2_5_dir_{name}", "us_aqi": f"us_aqi_dir_{name}"})
        merged = dir_df if merged is None else pd.merge(merged, dir_df, on="time", how="outer")
    return merged

# CLEANING
def clean_dataset(df: pd.DataFrame, extra_numeric_cols: list = None) -> pd.DataFrame:
    """Data-quality cleaning on the raw merged data, BEFORE feature
    engineering -- derived features (lags, rolling means) are computed from
    these values, so gaps/bad values here would otherwise propagate into
    every downstream feature."""
    print("\nCleaning dataset...")
    before = len(df)

    dup_count = df.duplicated(subset=["datetime"]).sum()
    if dup_count:
        df = df.drop_duplicates(subset=["datetime"], keep="first").reset_index(drop=True)
        print(f"  Removed {dup_count} duplicate datetime rows")

    full_range = pd.date_range(df["datetime"].min(), df["datetime"].max(),
                                freq="h", tz=df["datetime"].dt.tz)
    missing_hours = full_range.difference(df["datetime"])
    if len(missing_hours):
        print(f"  Note: {len(missing_hours)} hourly timestamps missing entirely "
              f"from the sequence (gaps in source data, not interpolated)")

    numeric_cols = WEATHER_VARS + AIR_QUALITY_VARS + (extra_numeric_cols or [])
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    missing_before = df[numeric_cols].isna().sum().sum()
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    missing_after = df[numeric_cols].isna().sum().sum()
    print(f"  Interpolated missing readings: {missing_before} -> {missing_after} remaining NaNs")

    clip_cols = ["precipitation", "pm10", "pm2_5", "carbon_monoxide",
                 "nitrogen_dioxide", "sulphur_dioxide", "ozone", "us_aqi"]
    clip_cols += [c for c in (extra_numeric_cols or []) if "pm2_5" in c or "us_aqi" in c]
    for col in clip_cols:
        if col not in df.columns:
            continue
        negative_count = (df[col] < 0).sum()
        if negative_count:
            df[col] = df[col].clip(lower=0)
            print(f"  Clipped {negative_count} negative value(s) in '{col}' to 0")

    print(f"Cleaning done: {before:,} -> {len(df):,} rows")
    return df

# FEATURE ENGINEERING
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["datetime"]
    df["hour"] = dt.dt.hour
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear
    return df

def add_aqi_features(df: pd.DataFrame, col: str = TARGET_COL, prefix: str = "aqi") -> pd.DataFrame:
    """Lag features at exactly the 3 forecast horizons, a short rolling mean
    for trend, and the AQI change rate."""
    for lag in LAG_HOURS:
        df[f"{prefix}_lag_{lag}h"] = df[col].shift(lag)
    df[f"{prefix}_rolling_mean_{ROLLING_WINDOWS[0]}h"] = df[col].rolling(ROLLING_WINDOWS[0]).mean()
    df[f"{prefix}_change_rate_1h"] = df[col].diff()
    return df

def add_pm25_change_rate(df: pd.DataFrame) -> pd.DataFrame:
    df["pm25_change_rate_1h"] = df["pm2_5"].diff()
    return df

def add_extra_lag_features(df: pd.DataFrame, col: str = TARGET_COL, prefix: str = "aqi") -> pd.DataFrame:
    """Additional AQI lags between the 3 forecast-horizon lags (6/12/36/60h)
    plus a rolling std -- volatility itself is signal (a calm, low-std recent
    window behaves very differently than one coming off a spike), which a
    rolling MEAN alone can't capture. All lags here are <= 60h, so none of
    these extend the max lookback beyond aqi_lag_72h -- no change needed to
    EDGE_SENSITIVE_COLS."""
    for lag in EXTRA_LAG_HOURS:
        df[f"{prefix}_lag_{lag}h"] = df[col].shift(lag)
    df[f"{prefix}_rolling_std_{ROLLING_WINDOWS[0]}h"] = df[col].rolling(ROLLING_WINDOWS[0]).std()
    return df

POLLUTANT_LAG_HOURS = [24, 48, 72]  # matches all 3 forecast horizons, not just 24h

def add_pollutant_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lags of ALL measured pollutants at 24h/48h/72h -- PM2.5 and NO2 were
    originally picked by correlation with same-hour AQI, but that ranking
    can shift once combined with horizon-matched lags; cheap to add the
    rest (co, so2, ozone, pm10) and let feature importance decide what's
    used. Lagging at 48h/72h too (not just 24h) matters because a
    pollutant's own multi-day memory is a different signal from AQI's
    multi-day memory -- e.g. NO2 (traffic-driven, decays faster) trending
    down over 72h while AQI stays flat tells the 72h model something AQI's
    own lag alone can't.These are PAST readings only (shift with a positive
    lag), not future --unlike add_future_weather_features(), pollutant 
    concentrations are direct inputs to the US AQI formula, so only past 
    values are safe to use here."""
    pollutants = {
        "pm25": "pm2_5", "no2": "nitrogen_dioxide", "pm10": "pm10",
        "co": "carbon_monoxide", "so2": "sulphur_dioxide", "ozone": "ozone",
    }
    for lag in POLLUTANT_LAG_HOURS:
        for short_name, col in pollutants.items():
            df[f"{short_name}_lag_{lag}h"] = df[col].shift(lag)
    return df

def add_pm_ratio_feature(df: pd.DataFrame) -> pd.DataFrame:
    """PM2.5/PM10 ratio -- a standard air-quality source signature. A LOW
    ratio (PM10 dominant) points to coarse dust/construction-type particles;
    a HIGH ratio (PM2.5 dominant) points to combustion sources (traffic,
    industrial burning). Karachi sees both regimes depending on wind and
    season, and the two have different persistence -- dust events clear
    faster than combustion haze -- so this ratio carries information
    neither pollutant's raw level does on its own. Small epsilon avoids
    div-by-zero on near-zero PM10 readings."""
    df["pm25_pm10_ratio"] = df["pm2_5"] / (df["pm10"] + 1e-3)
    return df

def add_weather_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling weather aggregates that capture stagnation/dispersion
    conditions rather than instantaneous readings -- sustained low wind
    speed lets pollutants accumulate, and recent rainfall scrubs the air,
    both physically meaningful drivers of AQI beyond what a single-hour
    snapshot conveys. Wind speed STD (not just mean) is added too: a
    calm, steady 24h and a 24h that swung from calm to gusty and back can
    share the same mean but represent very different dispersion histories.
    A 72h precipitation sum is added alongside the existing 24h one -- a
    dry 72h window is a materially different washout history than a dry
    24h window that follows heavy rain the day before.
    stagnation_index is an explicit interaction (low wind AND no rain =
    pollutants both accumulating and not being washed out), not just two
    separate columns. This matters specifically because Ridge -- the model
    that's been winning every horizon so far -- is linear and can't
    discover this kind of AND-combination on its own from the two raw
    columns; tree models could already infer it via splits on both, but
    Ridge needs it handed to it explicitly to use it at all.
    """
    df["wind_speed_rolling_mean_24h"] = df["wind_speed_10m"].rolling(24).mean()
    df["wind_speed_rolling_std_24h"] = df["wind_speed_10m"].rolling(24).std()
    df["precipitation_rolling_sum_24h"] = df["precipitation"].rolling(24).sum()
    df["precipitation_rolling_sum_72h"] = df["precipitation"].rolling(72).sum()
    df["stagnation_index"] = (
        1 / (df["wind_speed_rolling_mean_24h"] + 1)
    ) * (1 / (df["precipitation_rolling_sum_24h"] + 1))
    return df

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Day-of-week signal (traffic/industrial activity patterns) -- distinct
    from the existing raw 'day' (day-of-month) column, which carries little
    real signal for AQI and isn't cyclically encoded like hour/month are.
    is_rush_hour is a separate, sharper signal than hour_sin/cos: traffic
    emissions spike in two short windows (morning + evening commute)
    rather than varying smoothly across the day, which a single sin/cos
    pair can't represent as a step-like effect. Kept as an explicit 0/1
    flag mainly because tree-based models (RF/XGBoost) split on thresholds
    and benefit from an explicit flag more than from inferring it out of a
    continuous cyclical encoding."""
    df["is_weekend"] = df["datetime"].dt.dayofweek.isin([5, 6]).astype(int)
    hour = df["datetime"].dt.hour
    df["is_rush_hour"] = (hour.isin([7, 8, 9]) | hour.isin([17, 18, 19, 20])).astype(int)
    import holidays as holidays_lib
    years = sorted(df["datetime"].dt.year.unique().tolist())
    pk_holidays = holidays_lib.Pakistan(years=years)
    df["is_public_holiday"] = df["datetime"].dt.date.isin(pk_holidays).astype(int)
    return df

def add_aqi_acceleration_feature(df: pd.DataFrame, col: str = TARGET_COL) -> pd.DataFrame:
    """Second derivative of AQI -- is the rate of change itself speeding up
    or slowing down. aqi_change_rate_1h (already computed) says AQI is
    rising; this says whether it's rising FASTER than it was an hour ago,
    which is the difference between an AQI event that's accelerating into
    a spike versus one that's already leveling off -- relevant information
    a first derivative alone doesn't carry, especially for the 48h/72h
    horizons where knowing if a trend is decelerating matters for whether
    it'll still be elevated that far out."""
    df["aqi_acceleration_1h"] = df[col].diff().diff()
    return df

def add_aqi_rolling_extremes(df: pd.DataFrame, col: str = TARGET_COL, prefix: str = "aqi") -> pd.DataFrame:
    """Rolling min/max over the same 24h window as the existing rolling
    mean/std -- 'how bad has it peaked recently' and 'how clean was the
    best hour recently' are both information a mean/std pair alone doesn't
    carry (e.g. a volatile day and a steadily worsening day can share the
    same mean+std but have very different max)."""
    df[f"{prefix}_rolling_max_{ROLLING_WINDOWS[0]}h"] = df[col].rolling(ROLLING_WINDOWS[0]).max()
    df[f"{prefix}_rolling_min_{ROLLING_WINDOWS[0]}h"] = df[col].rolling(ROLLING_WINDOWS[0]).min()
    return df

MULTI_HORIZON_ROLLING_HOURS = [48, 72]  # matches the 48h/72h forecast horizons directly

def add_long_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling means for AQI and its two most predictive pollutants at
    multiple windows: 48h and 72h (matching the forecast horizons
    themselves -- the 48h model gets to see 'what's the 48h trend',
    the 72h model 'what's the 72h trend'), plus a 7-day (168h) window for
    slower-moving background trend. The existing 24h rolling window
    captures short-term trend well but has mostly decayed out by 48-72h
    ahead -- these give the longer-horizon models trend signal that's
    still informative that far out."""
    for window in MULTI_HORIZON_ROLLING_HOURS:
        df[f"aqi_rolling_mean_{window}h"] = df[TARGET_COL].rolling(window).mean()
    for window in MULTI_HORIZON_ROLLING_HOURS + [LONG_ROLLING_WINDOW_HOURS]:
        df[f"pm2_5_rolling_mean_{window}h"] = df["pm2_5"].rolling(window).mean()
        df[f"pm10_rolling_mean_{window}h"] = df["pm10"].rolling(window).mean()
    df[f"aqi_rolling_mean_{LONG_ROLLING_WINDOW_HOURS}h"] = (
        df[TARGET_COL].rolling(LONG_ROLLING_WINDOW_HOURS).mean()
    )
    return df

def add_weather_tendency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Short-term rate of change, not just level -- falling pressure often
    signals an approaching front that disperses pollution, while a flat/
    rising pressure trend is associated with the stagnant high-pressure
    conditions that let AQI build up. 3h is short enough to be a leading
    indicator rather than duplicating the 24h rolling features."""
    df["pressure_msl_change_3h"] = df["pressure_msl"].diff(3)
    df["temperature_2m_change_3h"] = df["temperature_2m"].diff(3)
    return df

def add_upwind_feature(df: pd.DataFrame, bearings: dict = UPWIND_BEARINGS) -> pd.DataFrame:
    """Blends the per-bearing upwind AQI/PM2.5 points (already merged into
    df as pm2_5_dir_{name}/us_aqi_dir_{name} columns during fetch) into a
    single wind-direction-weighted 'what's approaching' feature.
    wind_direction_10m is meteorological convention: the direction the wind
    is blowing FROM. So the fixed point whose bearing from the city best
    matches the CURRENT wind direction is the most "upwind" one right now
    -- weight it highest. A calm/zero wind produces near-equal weights
    across all bearings (no directional information available), handled by
    the epsilon-padded normalization below."""
    wind_dir_rad = np.radians(df["wind_direction_10m"])
    weights = {}
    for name, bearing in bearings.items():
        # cos(0) = 1 when this bearing exactly matches the wind's source
        # direction, falling to 0 (clipped) for bearings >= 90 deg away.
        weights[name] = np.clip(np.cos(wind_dir_rad - np.radians(bearing)), 0, None)
    weight_sum = sum(weights.values()) + 1e-6  # avoid div-by-zero on calm wind
    df["aqi_upwind"] = sum(
        weights[name] / weight_sum * df[f"us_aqi_dir_{name}"] for name in bearings
    )
    df["pm25_upwind"] = sum(
        weights[name] / weight_sum * df[f"pm2_5_dir_{name}"] for name in bearings
    )
    return df

def add_forecast_targets(df: pd.DataFrame, col: str, horizons_hours=FORECAST_HORIZONS) -> pd.DataFrame:
    for h in horizons_hours:
        df[f"target_{col}_{h}h_ahead"] = df[col].shift(-h)
    return df

CYCLICAL_PERIODS = {"hour": 24, "month": 12, "day_of_year": 366}
WIND_DIR_COLS_TO_ENCODE = ["wind_direction_10m"] + [
    f"wind_direction_10m_future_{h}h" for h in FORECAST_HORIZONS
]

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes periodic columns (hour-of-day, month-of-year, wind direction
    in degrees) as sin/cos pairs and drops the raw column.
    Raw integer/degree values misrepresent distance for anything that
    wraps around: hour 23 and hour 0 are ~23 apart numerically but 1 hour
    apart in reality -- same problem for Dec->Jan (month 12 vs 1) and wind
    direction wrapping at 360->0. This mostly matters for Ridge and the
    neural net, which act on raw feature values/distances directly;
    tree-based models (RF/XGBoost) split on thresholds and are largely
    unaffected, which is part of why they weren't visibly hurt by this
    before.
    """
    for col, period in CYCLICAL_PERIODS.items():
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
        df = df.drop(columns=[col])

    for col in WIND_DIR_COLS_TO_ENCODE:
        if col in df.columns:
            df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / 360)
            df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / 360)
            df = df.drop(columns=[col])
    return df

LOW_VALUE_FEATURES = [
    "precipitation", "is_rush_hour", "temperature_2m", "pressure_msl_change_3h",
    "ozone_lag_48h", "co_lag_24h", "wind_speed_10m", "ozone_lag_72h",
    "relative_humidity_2m", "pm25_change_rate_1h", "ozone_lag_24h", "hour_cos",
]

def drop_low_value_features(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in LOW_VALUE_FEATURES if c in df.columns]
    if present:
        df = df.drop(columns=present)
        print(f"  Dropped {len(present)} low-value features (verified via CV, see LOW_VALUE_FEATURES): {present}")
    return df

# HOPSWORKS: UPLOAD + VERIFY
def delete_existing_csv(path: str):
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted existing {path} before writing the new one")

def get_hopsworks_feature_store():
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "HOPSWORKS_API_KEY not found. Add a .env file in this folder with "
            "a line: HOPSWORKS_API_KEY=your_key_here"
        )
    print("Logging into Hopsworks...")
    project = hopsworks.login(api_key_value=api_key)
    return project.get_feature_store()

def delete_existing_feature_views(fs, city: str):
    """Feature Views must be deleted BEFORE their underlying feature group --
    Hopsworks blocks feature group deletion while a dependent Feature View
    exists. Without this, the feature group delete silently fails and data
    gets upserted into the old group instead of a clean rebuild, and the
    Feature View version keeps incrementing instead of staying at 1."""
    try:
        fvs = fs.get_feature_views(name=f"aqi_fv_{city}")
    except Exception:
        fvs = []
    for fv in fvs:
        print(f"Deleting feature view 'aqi_fv_{city}' v{fv.version}...")
        fv.delete()
    if fvs:
        print(f"Deleted {len(fvs)} existing feature view(s).")
    else:
        print("No existing feature views to delete.")

def delete_existing_feature_group(fs, city: str, version: int = 1):
    """Deletes the existing feature group (if any) before this run recreates
    it -- keeps version locked at 1 instead of accumulating versions or
    silently upserting into a stale group."""
    try:
        fg = fs.get_feature_group(f"aqi_features_{city}", version=version)
    except Exception:
        print(f"No existing feature group 'aqi_features_{city}' v{version} found -- nothing to delete")
        return
    try:
        print(f"Found existing feature group 'aqi_features_{city}' v{version}, deleting...")
        fg.delete()
        print("Deleted.")
    except Exception as e:
        # This is a REAL failure, not a "nothing to delete" case -- surface
        # it clearly instead of masking it as if nothing was wrong.
        print(f"WARNING: found the feature group but FAILED to delete it: {e}")
        print("The run will continue, but data may get upserted into the "
              "existing group instead of a clean rebuild.")

def upload_to_hopsworks(fs, df: pd.DataFrame, city: str, version: int = 1):
    df = df.copy()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    for col in ["day"]:
        if col in df.columns:
            df[col] = df[col].astype("int32")

    delete_existing_feature_views(fs, city)
    delete_existing_feature_group(fs, city, version)
    fg = fs.get_or_create_feature_group(
        name=f"aqi_features_{city}",
        version=version,
        description=f"Hourly weather + air quality raw + engineered features for {city}",
        primary_key=["datetime"],
        event_time="datetime",
        online_enabled=True,
        time_travel_format="HUDI",
    )
    print(f"Inserting {len(df):,} rows into feature group 'aqi_features_{city}' v{version}...")
    fg.insert(df)
    return version

def verify_upload(fs, city: str, version: int, expected_rows: int,
                   max_wait_seconds: int = 600, initial_wait_seconds: int = 45):
    """Poll until the offline materialization job finishes, then read back
    and verify. Reading too early raises a 'no Hudi commits found' error
    even though the upload succeeded -- expected, not a real failure."""
    fg = fs.get_feature_group(f"aqi_features_{city}", version=version)
    print(f"Giving materialization a {initial_wait_seconds}s head start before checking...")
    time.sleep(initial_wait_seconds)
    waited = initial_wait_seconds
    poll_interval = 20
    not_ready_markers = ("hudi properties", "no data has been written",
                          "hudi commits", "query service")
    while waited < max_wait_seconds:
        try:
            df_check = fg.read()
            break
        except Exception as e:
            if any(m in str(e).lower() for m in not_ready_markers):
                print(f"  Still materializing, waiting... ({waited}s elapsed)")
                time.sleep(poll_interval)
                waited += poll_interval
            else:
                raise
    else:
        print(f"Timed out after {max_wait_seconds}s waiting for materialization. "
              f"Check the job status in the Hopsworks UI.")
        return None

    print(f"Read back: {df_check.shape}")
    print(f"Date range: {df_check['datetime'].min()} -> {df_check['datetime'].max()}")
    if len(df_check) == expected_rows:
        print(f"MATCH: {expected_rows:,} rows uploaded == {len(df_check):,} rows read back")
    else:
        print(f"MISMATCH: uploaded {expected_rows:,} rows but read back {len(df_check):,}")
    return df_check

# TRAIN/TEST SPLIT
def prepare_for_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("datetime").reset_index(drop=True)
    before = len(df)
    df_clean = df.dropna(subset=EDGE_SENSITIVE_COLS).reset_index(drop=True)
    print(f"Dropped {before - len(df_clean):,} edge rows without full lag/target "
          f"history -- {len(df_clean):,} rows usable for training/evaluation")
    return df_clean

def compute_split_boundaries(df: pd.DataFrame) -> dict:
    last_date = df["datetime"].iloc[-1]
    test_start = last_date - pd.DateOffset(months=12) + pd.Timedelta(hours=1)
    train_mask = df["datetime"] < test_start
    test_mask = df["datetime"] >= test_start
    bounds = {
        "train_start": df.loc[train_mask, "datetime"].iloc[0],
        "train_end": df.loc[train_mask, "datetime"].iloc[-1],
        "test_start": df.loc[test_mask, "datetime"].iloc[0],
        "test_end": df.loc[test_mask, "datetime"].iloc[-1],
    }
    n, n_train, n_test = len(df), train_mask.sum(), test_mask.sum()
    print(f"\nSplit boundaries (12-month test, calendar-anchored, no separate val):")
    print(f"  Train: {bounds['train_start']} -> {bounds['train_end']}  ({n_train:,} rows, {n_train/n:.0%})")
    print(f"  Test:  {bounds['test_start']} -> {bounds['test_end']}  ({n_test:,} rows, {n_test/n:.0%})")
    return bounds

def save_local_split_csvs(df: pd.DataFrame, bounds: dict):
    """Local-only fallback, used ONLY when --skip-hopsworks is set."""
    train_df = df[(df["datetime"] >= bounds["train_start"]) & (df["datetime"] <= bounds["train_end"])]
    test_df = df[(df["datetime"] >= bounds["test_start"]) & (df["datetime"] <= bounds["test_end"])]
    for name, part in [("train.csv", train_df), ("test.csv", test_df)]:
        delete_existing_csv(name)
        part.to_csv(name, index=False)

    print(f"\nSaved local split CSVs: train.csv ({len(train_df):,} rows), test.csv ({len(test_df):,} rows)")
    return train_df, test_df

def fetch_split_from_hopsworks(fv, training_dataset_version: int = 1):
    print("\nFetching the actual split data back from Hopsworks (single source of truth)...")
    X_train, X_test, y_train, y_test = fv.get_train_test_split(
        training_dataset_version=training_dataset_version
    )
    train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    for name, part in [("train", train_df), ("test", test_df)]:
        if "datetime" not in part.columns:
            raise KeyError(f"'datetime' missing from {name} split.")

    train_df = train_df.sort_values("datetime").reset_index(drop=True)
    test_df = test_df.sort_values("datetime").reset_index(drop=True)
    for name, part in [("train.csv", train_df), ("test.csv", test_df)]:
        delete_existing_csv(name)
        part.to_csv(name, index=False)

    print(f"Saved local CSVs from Hopsworks' split (re-sorted chronologically): "
          f"train.csv ({len(train_df):,} rows), test.csv ({len(test_df):,} rows)")
    return train_df, test_df

def create_feature_view_split(fs, df: pd.DataFrame, city: str, bounds: dict, version: int = 1):
    fg = fs.get_feature_group(f"aqi_features_{city}", version=version)
    query = fg.select(df.columns.tolist())
    fv = fs.get_or_create_feature_view(
        name=f"aqi_fv_{city}",
        version=version,
        query=query,
        labels=TARGET_COLS,
        description=f"3-day-ahead AQI forecasting feature view for {city}",
    )

    try:
        fv.delete_all_training_datasets()
        print("  Deleted all existing training dataset(s) via delete_all_training_datasets().")
    except AttributeError:
        try:
            existing_tds = fv.get_training_datasets()
            for td in existing_tds:
                print(f"  Deleting existing training dataset v{td.version} before creating a new split...")
                fv.delete_training_dataset(training_dataset_version=td.version)
        except Exception as e:
            print(f"  Could not enumerate/delete existing training datasets ({e}) -- "
                  f"proceeding to create the new split anyway. If old versions "
                  f"keep accumulating, check the Hopsworks UI (Feature View -> "
                  f"Training Datasets) and this hsfs version's delete API.")
    except Exception as e:
        print(f"  Could not delete existing training datasets ({e}) -- "
              f"proceeding to create the new split anyway. If old versions "
              f"keep accumulating, check the Hopsworks UI (Feature View -> "
              f"Training Datasets) and this hsfs version's delete API.")

    print("\nCreating train/test split in Hopsworks (event-time based, no separate val)...")
    fv.create_train_test_split(
        train_start=bounds["train_start"],
        train_end=bounds["train_end"],
        test_start=bounds["test_start"],
        test_end=bounds["test_end"],
        description=f"Calendar-anchored train/test split for {city}",
        data_format="csv",
        statistics_config=False,
    )
    print(f"Feature View 'aqi_fv_{city}' v{version} created with the split registered.")
    return fv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="Karachi")
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--years-back", type=int, default=4,
                         help="How many years of history to pull, ending today")
    parser.add_argument("--out", default="aqi_features.csv",
                         help="Single combined output CSV (raw + engineered columns)")
    parser.add_argument("--version", type=int, default=1,
                         help="Feature group / feature view version (stays fixed at 1)")
    parser.add_argument("--skip-hopsworks", action="store_true",
                         help="Only fetch + clean + engineer + split locally, no Hopsworks calls")
    parser.add_argument("--skip-upwind", action="store_true",
                         help="Skip the 4 extra upwind air-quality fetches (~5x air-quality fetch "
                              "time) -- useful for fast local iteration")
    parser.add_argument("--skip-feature-view", action="store_true",
                         help="Do everything except registering the Hopsworks Feature View split")
    parser.add_argument("--max-wait", type=int, default=600,
                         help="Max seconds to wait for materialization before giving up")
    parser.add_argument("--initial-wait", type=int, default=45,
                         help="Seconds to wait before the first read-back attempt")
    args = parser.parse_args()

    # Automatic date range: today going back N years
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=365 * args.years_back)).strftime("%Y-%m-%d")
    print(f"Date range: {start_date} -> {end_date} ({args.years_back} years back from today)")
    if args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
    else:
        lat, lon = geocode_city(args.city)

    # Fetch (Feature Pipeline requirement: raw weather + pollutant data)
    weather_df = fetch_weather(lat, lon, start_date, end_date)
    aq_df = fetch_air_quality(lat, lon, start_date, end_date)
    df = pd.merge(weather_df, aq_df, on="time", how="inner")
    upwind_cols = []
    if not args.skip_upwind:
        upwind_df = fetch_upwind_air_quality(lat, lon, start_date, end_date)
        upwind_cols = [c for c in upwind_df.columns if c != "time"]
        df = pd.merge(df, upwind_df, on="time", how="left")
    else:
        print("Skipping upwind air-quality fetch (--skip-upwind was set) -- "
              "aqi_upwind/pm25_upwind features will not be available")

    df = df.rename(columns={"time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"Fetched and merged {len(df):,} raw rows")

    # Clean 
    df = clean_dataset(df, extra_numeric_cols=upwind_cols)

    # Feature engineering (Feature Pipeline requirement: time-based + derived features like AQI change rate)
    df = add_time_features(df)
    df = add_aqi_features(df)
    df = add_aqi_acceleration_feature(df)
    df = add_pm25_change_rate(df)
    df = add_extra_lag_features(df)
    df = add_aqi_rolling_extremes(df)
    df = add_long_rolling_features(df)
    df = add_pollutant_lag_features(df)
    df = add_pm_ratio_feature(df)
    df = add_weather_trend_features(df)
    df = add_weather_tendency_features(df)
    df = add_calendar_features(df)
    if not args.skip_upwind:
        df = add_upwind_feature(df)
    df = add_forecast_targets(df, TARGET_COL, horizons_hours=FORECAST_HORIZONS)
    df = add_cyclical_features(df)
    df = drop_low_value_features(df)
    print(f"Feature engineering done: {df.shape[0]:,} rows x {df.shape[1]} columns")
    delete_existing_csv(args.out)
    df.to_csv(args.out, index=False)
    print(f"Saved combined raw+features CSV to {args.out}")

    city_key = args.city.lower().replace(" ", "_")

    if args.skip_hopsworks:
        print("Skipping Hopsworks upload (--skip-hopsworks was set) -- splitting locally only")
        df_for_split = prepare_for_split(df)
        bounds = compute_split_boundaries(df_for_split)
        save_local_split_csvs(df_for_split, bounds)
        return

    # Store in Feature Store (Feature Pipeline requirement)
    fs = get_hopsworks_feature_store()
    version = upload_to_hopsworks(fs, df, city_key, args.version)

    # Verify (Historical Data Backfill requirement:confirm feature pipeline actually produced usable training data)
    df_check = verify_upload(fs, city_key, version, expected_rows=len(df),
                              max_wait_seconds=args.max_wait,
                              initial_wait_seconds=args.initial_wait)
    if df_check is None:
        print("Could not verify materialization in time -- run the split step "
              "again later once the Hopsworks job finishes.")
        return

    # Historical Data Backfill requirement
    df_for_split = prepare_for_split(df_check)
    bounds = compute_split_boundaries(df_for_split)  # still needed: tells

    if args.skip_feature_view:
        print("Skipping Feature View creation (--skip-feature-view was set) "
              "-- falling back to a local pandas split")
        save_local_split_csvs(df_for_split, bounds)
        return

    fv = create_feature_view_split(fs, df_for_split, city_key, bounds, version=args.version)
    fetch_split_from_hopsworks(fv, training_dataset_version=1)

if __name__ == "__main__":
    main()

# conda create --name hopsworks_env python=3.10 -y
# conda activate hopsworks_env
# conda install -c conda-forge twofish -y
# pip install dotenv, hopsworks, pandas, requests, pyarrow, confluent-kafka, holidays
# python -m pip install --upgrade pip setuptools wheel
# python aqiPipeline.py --city "Karachi" --version 1
