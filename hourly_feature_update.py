"""
hourly_feature_update.py
Incremental (hourly) counterpart to aqiPipeline.py.

aqiPipeline.py does a FULL rebuild (delete + re-fetch N years + recreate
the feature group) -- that's the right tool for the initial backfill, but
far too slow/wasteful to run every hour. This script instead:

  1. Finds the latest timestamp already stored in the Feature Store.
  2. Fetches only a short recent window (default: last 10 days) -- enough
     to correctly warm up all lag/rolling features (max lookback used
     anywhere in the pipeline is 168h = 7 days), not the full history.
  3. Re-runs the SAME feature engineering functions from aqiPipeline.py
     (imported, not duplicated) over that short window.
  4. Inserts ONLY the rows newer than what's already stored (an upsert on
     the "datetime" primary key -- Hopsworks/Hudi handles this as a pure
     append since these datetimes don't exist yet).
  5. Prunes rows older than --retention-years (default 4) from the
     Feature Store, and mirrors the same append+prune behavior on the
     local aqi_features.csv, so both stay a fixed rolling window instead
     of growing forever.

This file does not modify aqiPipeline.py or train_models.py -- it only
imports read-only from aqiPipeline.py so feature engineering never drifts
out of sync between the two.

Meant to be run hourly by GitHub Actions -- see
.github/workflows/feature_pipeline.yml

Usage:
    python hourly_feature_update.py --city "Karachi"
"""
import argparse
import os
from datetime import datetime, timedelta

import pandas as pd

from aqiPipeline import (  # noqa: F401  -- read-only reuse, see module docstring
    geocode_city, fetch_weather, fetch_air_quality, fetch_upwind_air_quality,
    clean_dataset, add_time_features, add_aqi_features, add_aqi_acceleration_feature,
    add_pm25_change_rate, add_extra_lag_features, add_aqi_rolling_extremes,
    add_long_rolling_features, add_pollutant_lag_features, add_pm_ratio_feature,
    add_weather_trend_features, add_weather_tendency_features, add_calendar_features,
    add_upwind_feature, add_forecast_targets, add_cyclical_features, drop_low_value_features,
    get_hopsworks_feature_store, delete_existing_csv,
    TARGET_COL, FORECAST_HORIZONS, LONG_ROLLING_WINDOW_HOURS,
)

# Long enough to fully warm up every lag/rolling feature (max lookback
# anywhere in the pipeline is 168h = 7 days) with margin to spare.
LOOKBACK_BUFFER_DAYS = 10
RETENTION_YEARS = 4
FEATURE_GROUP_VERSION = 1


# ==================================================================
# 1. FIND WHAT'S ALREADY STORED
# ==================================================================

def get_feature_group(fs, city: str, version: int = FEATURE_GROUP_VERSION):
    return fs.get_feature_group(f"aqi_features_{city}", version=version)


def get_latest_stored_datetime(fs, city: str):
    """Reads only a recent slice (not the whole 4-year history) to find the
    most recent stored row -- fast, avoids pulling everything just to find
    the tail. Returns None if the feature group is empty/missing (cold
    start -- shouldn't normally happen if aqiPipeline.py already backfilled
    it, but handled defensively)."""
    fg = get_feature_group(fs, city)
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - timedelta(days=LOOKBACK_BUFFER_DAYS + 2)
    try:
        df = fg.filter(fg.datetime >= cutoff.strftime("%Y-%m-%d %H:%M:%S")).read()
    except Exception as e:
        print(f"  Filtered read failed ({e}), falling back to a full read...")
        df = fg.read()

    if df is None or df.empty:
        print("  Feature group has no rows in the recent window -- treating as cold start.")
        return None

    df["datetime"] = pd.to_datetime(df["datetime"])
    latest = df["datetime"].max()
    # Hopsworks/Hudi returns this column as tz-aware (UTC-labeled) on read,
    # even though everything aqiPipeline.py writes/compares is tz-naive
    # (Open-Meteo's "timezone": "auto" gives naive local timestamps). The
    # underlying values aren't shifted -- this is a labeling mismatch, not
    # a real offset -- but pandas refuses to compare naive vs. aware
    # datetimes at all, so normalize to naive here, once, at the source.
    if latest.tzinfo is not None:
        latest = latest.tz_localize(None)
    print(f"  Latest stored row: {latest}")
    return latest


# ==================================================================
# 2. FETCH + ENGINEER A SHORT RECENT WINDOW
# ==================================================================

def fetch_recent_window(lat: float, lon: float, start_date: str, end_date: str, skip_upwind: bool):
    print(f"Fetching recent window: {start_date} -> {end_date}")
    weather_df = fetch_weather(lat, lon, start_date, end_date)
    aq_df = fetch_air_quality(lat, lon, start_date, end_date)
    df = pd.merge(weather_df, aq_df, on="time", how="inner")

    upwind_cols = []
    if not skip_upwind:
        upwind_df = fetch_upwind_air_quality(lat, lon, start_date, end_date)
        upwind_cols = [c for c in upwind_df.columns if c != "time"]
        df = pd.merge(df, upwind_df, on="time", how="left")

    df = df.rename(columns={"time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"  Fetched {len(df):,} raw rows")
    return df, upwind_cols


def engineer_features(df: pd.DataFrame, upwind_cols: list, skip_upwind: bool) -> pd.DataFrame:
    """Same function sequence, same order, as aqiPipeline.py's main() --
    kept side by side with that file on purpose so a diff between the two
    call sequences is easy to spot if aqiPipeline.py's sequence ever
    changes."""
    df = clean_dataset(df, extra_numeric_cols=upwind_cols)
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
    if not skip_upwind:
        df = add_upwind_feature(df)
    df = add_forecast_targets(df, TARGET_COL, horizons_hours=FORECAST_HORIZONS)
    df = add_cyclical_features(df)
    df = drop_low_value_features(df)

    # Rows at the START of this short window won't have a full 168h of
    # lookback WITHIN the window itself (even though the real history
    # exists in the Feature Store) -- drop those warm-up rows rather than
    # upload partially-computed features. By the time we reach the newest
    # rows (the ones we actually care about), LOOKBACK_BUFFER_DAYS is
    # comfortably past the 168h warm-up point.
    warmup_col = f"aqi_rolling_mean_{LONG_ROLLING_WINDOW_HOURS}h"
    before = len(df)
    df = df.dropna(subset=[warmup_col]).reset_index(drop=True)
    print(f"  Dropped {before - len(df):,} warm-up rows without a full {LONG_ROLLING_WINDOW_HOURS}h lookback")
    return df


# ==================================================================
# 3. INSERT NEW ROWS
# ==================================================================

def select_new_rows(engineered_df: pd.DataFrame, latest_stored_dt) -> pd.DataFrame:
    if latest_stored_dt is None:
        return engineered_df
    # Defensive: normalize both sides to naive regardless of what arrives
    # here, so this never breaks again on a tz-labeling mismatch.
    if getattr(latest_stored_dt, "tzinfo", None) is not None:
        latest_stored_dt = latest_stored_dt.tz_localize(None)
    dt_col = engineered_df["datetime"]
    if hasattr(dt_col.dt, "tz") and dt_col.dt.tz is not None:
        dt_col = dt_col.dt.tz_localize(None)
    return engineered_df[dt_col > latest_stored_dt].reset_index(drop=True)


def align_dtypes_to_feature_group(df: pd.DataFrame, fg) -> pd.DataFrame:
    """Casts numeric columns to match the Feature Group's EXISTING schema
    exactly, read live from Hopsworks rather than assumed.

    Why this is needed: pandas'/numpy's default integer dtype is
    platform-dependent -- int64 on Linux/Mac, int32 on Windows. The
    feature group's schema was locked in by whichever platform ran the
    very first insert (via aqiPipeline.py). Any later insert -- like this
    hourly one -- run from a DIFFERENT platform can produce a different
    default int width for the exact same column, which Hopsworks rejects
    as a schema mismatch even though the actual data is fine. Reading the
    schema back and casting to match it sidesteps this regardless of which
    OS either script happens to run on."""
    type_map = {
        "bigint": "int64", "int": "int32", "smallint": "int16", "tinyint": "int8",
        "float": "float32", "double": "float64",
    }
    schema = {f.name: f.type for f in fg.features}
    df = df.copy()
    for col in df.columns:
        target_dtype = type_map.get(schema.get(col))
        if target_dtype:
            try:
                df[col] = df[col].astype(target_dtype)
            except (ValueError, TypeError) as e:
                print(f"  Could not cast '{col}' to {target_dtype} (Hopsworks type: {schema.get(col)}): {e}")
    return df


def insert_new_rows(fg, new_rows: pd.DataFrame):
    if new_rows.empty:
        print("No new rows to insert (source data hasn't advanced past the latest stored hour yet).")
        return
    df = new_rows.copy()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    try:
        df = align_dtypes_to_feature_group(df, fg)
    except Exception as e:
        # Defensive fallback if schema introspection itself fails for some
        # reason -- at minimum keep the one cast aqiPipeline.py's own
        # upload path relies on.
        print(f"  Schema introspection failed ({e}), falling back to a manual cast for 'day' only")
        if "day" in df.columns:
            df["day"] = df["day"].astype("int32")

    print(f"Inserting {len(df):,} new row(s) -- {df['datetime'].min()} -> {df['datetime'].max()}")
    fg.insert(df)


# ==================================================================
# 4. PRUNE: KEEP ONLY THE LATEST N YEARS
# ==================================================================

def prune_old_rows(fg, retention_years: int = RETENTION_YEARS):
    """Deletes rows older than the retention window from the Feature Store
    so it stays a fixed rolling window instead of growing forever. Uses
    HSFS's commit_delete() on this Hudi-backed feature group (aqiPipeline.py
    creates it with time_travel_format="HUDI"), which needs a DataFrame of
    the rows to remove (matched by primary key + event time)."""
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=retention_years)
    print(f"Pruning rows older than {cutoff.date()} (keeping latest {retention_years} years)...")
    try:
        old_df = fg.filter(fg.datetime < cutoff.strftime("%Y-%m-%d %H:%M:%S")).read()
    except Exception as e:
        print(f"  Could not read old rows for pruning, skipping this run's prune step: {e}")
        return

    if old_df is None or old_df.empty:
        print("  No rows older than the retention window -- nothing to prune.")
        return

    old_df["datetime"] = pd.to_datetime(old_df["datetime"])
    print(f"  Deleting {len(old_df):,} rows ({old_df['datetime'].min()} -> {old_df['datetime'].max()})...")
    try:
        fg.commit_delete(old_df)
        print("  Pruned.")
    except Exception as e:
        print(f"  WARNING: commit_delete failed ({e}). If your installed hopsworks/hsfs "
              f"version exposes a different row-delete API, this call needs updating -- "
              f"check the Hopsworks docs for your version's Feature Group delete method.")


# ==================================================================
# 5. MIRROR THE SAME APPEND + PRUNE ON THE LOCAL CSV
# ==================================================================

def update_local_csv(new_rows: pd.DataFrame, csv_path: str, retention_years: int = RETENTION_YEARS):
    if new_rows.empty and not os.path.exists(csv_path):
        return
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path, parse_dates=["datetime"])
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=["datetime"], keep="last")
    else:
        combined = new_rows.copy()

    cutoff = pd.Timestamp.now() - pd.DateOffset(years=retention_years)
    before = len(combined)
    combined = combined[combined["datetime"] >= cutoff]
    combined = combined.sort_values("datetime").reset_index(drop=True)

    delete_existing_csv(csv_path)
    combined.to_csv(csv_path, index=False)
    print(f"Updated {csv_path}: {before:,} -> {len(combined):,} rows after retention trim "
          f"({combined['datetime'].min()} -> {combined['datetime'].max()})")


# ==================================================================
# MAIN
# ==================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="Karachi")
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--out", default="aqi_features.csv")
    parser.add_argument("--retention-years", type=int, default=RETENTION_YEARS)
    parser.add_argument("--lookback-days", type=int, default=LOOKBACK_BUFFER_DAYS)
    parser.add_argument("--skip-upwind", action="store_true")
    parser.add_argument("--skip-prune", action="store_true",
                         help="Insert new rows but don't prune old ones this run")
    args = parser.parse_args()

    if args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
    else:
        lat, lon = geocode_city(args.city)
    city_key = args.city.lower().replace(" ", "_")

    fs = get_hopsworks_feature_store()
    fg = get_feature_group(fs, city_key)

    print("\nChecking latest stored row...")
    latest_stored_dt = get_latest_stored_datetime(fs, city_key)

    end_date = datetime.utcnow().strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=args.lookback_days)).strftime("%Y-%m-%d")

    raw_df, upwind_cols = fetch_recent_window(lat, lon, start_date, end_date, args.skip_upwind)
    engineered_df = engineer_features(raw_df, upwind_cols, args.skip_upwind)

    new_rows = select_new_rows(engineered_df, latest_stored_dt)
    insert_new_rows(fg, new_rows)
    update_local_csv(new_rows, args.out, args.retention_years)

    if not args.skip_prune:
        prune_old_rows(fg, args.retention_years)
    else:
        print("Skipping prune step (--skip-prune was set)")

    print("\nHourly update complete.")


if __name__ == "__main__":
    main()
