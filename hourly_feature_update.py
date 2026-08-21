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
     Feature Store via explicit key deletion (Option 1) to avoid breaking
     downstream Feature Views, and mirrors the same append+prune behavior
     on the local aqi_features.csv.

Usage:
    python hourly_feature_update.py --city "Karachi"
"""

import argparse
import os
from datetime import datetime, timedelta

import pandas as pd

from aqiPipeline import (  # noqa: F401 -- read-only reuse, see module docstring
    geocode_city,
    fetch_weather,
    fetch_air_quality,
    fetch_upwind_air_quality,
    clean_dataset,
    add_time_features,
    add_aqi_features,
    add_aqi_acceleration_feature,
    add_pm25_change_rate,
    add_extra_lag_features,
    add_aqi_rolling_extremes,
    add_long_rolling_features,
    add_pollutant_lag_features,
    add_pm_ratio_feature,
    add_weather_trend_features,
    add_weather_tendency_features,
    add_calendar_features,
    add_upwind_feature,
    add_forecast_targets,
    add_cyclical_features,
    drop_low_value_features,
    get_hopsworks_feature_store,
    delete_existing_csv,
    TARGET_COL,
    FORECAST_HORIZONS,
    LONG_ROLLING_WINDOW_HOURS,
)


# ============================================================================
# Configuration constants
# ============================================================================

LOOKBACK_BUFFER_DAYS = 10
RETENTION_YEARS = 4
FEATURE_GROUP_VERSION = 1


# ============================================================================
# 1. FIND WHAT'S ALREADY STORED
# ============================================================================

def get_feature_group(
    fs,
    city: str,
    version: int = FEATURE_GROUP_VERSION,
):
    return fs.get_feature_group(
        f"aqi_features_{city}",
        version=version,
    )


def get_latest_stored_datetime(fs, city: str):
    """Reads only a recent slice to find the most recent stored row."""

    fg = get_feature_group(fs, city)

    cutoff = (
        pd.Timestamp.utcnow().tz_localize(None)
        - timedelta(days=LOOKBACK_BUFFER_DAYS + 2)
    )

    try:
        df = fg.filter(
            fg.datetime >= cutoff.strftime("%Y-%m-%d %H:%M:%S")
        ).read()

    except Exception as e:
        print(
            f"  Filtered read failed ({e}), "
            "falling back to a full read..."
        )
        df = fg.read()

    if df is None or df.empty:
        print(
            "  Feature group has no rows in the recent window "
            "-- treating as cold start."
        )
        return None

    df["datetime"] = pd.to_datetime(df["datetime"])

    latest = df["datetime"].max()

    if latest.tzinfo is not None:
        latest = latest.tz_localize(None)

    print(f"  Latest stored row: {latest}")

    return latest


# ============================================================================
# 2. FETCH + ENGINEER A SHORT RECENT WINDOW
# ============================================================================

def fetch_recent_window(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    skip_upwind: bool,
):
    print(f"Fetching recent window: {start_date} -> {end_date}")

    weather_df = fetch_weather(
        lat,
        lon,
        start_date,
        end_date,
    )

    aq_df = fetch_air_quality(
        lat,
        lon,
        start_date,
        end_date,
    )

    df = pd.merge(
        weather_df,
        aq_df,
        on="time",
        how="inner",
    )

    upwind_cols = []

    if not skip_upwind:
        upwind_df = fetch_upwind_air_quality(
            lat,
            lon,
            start_date,
            end_date,
        )

        upwind_cols = [
            c for c in upwind_df.columns
            if c != "time"
        ]

        df = pd.merge(
            df,
            upwind_df,
            on="time",
            how="left",
        )

    df = df.rename(
        columns={"time": "datetime"}
    )

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    df = (
        df.sort_values("datetime")
        .reset_index(drop=True)
    )

    print(f"  Fetched {len(df):,} raw rows")

    return df, upwind_cols


def engineer_features(
    df: pd.DataFrame,
    upwind_cols: list,
    skip_upwind: bool,
) -> pd.DataFrame:
    """Same function sequence, same order, as aqiPipeline.py's main()."""

    df = clean_dataset(
        df,
        extra_numeric_cols=upwind_cols,
    )

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

    df = add_forecast_targets(
        df,
        TARGET_COL,
        horizons_hours=FORECAST_HORIZONS,
    )

    df = add_cyclical_features(df)
    df = drop_low_value_features(df)

    warmup_col = (
        f"aqi_rolling_mean_{LONG_ROLLING_WINDOW_HOURS}h"
    )

    before = len(df)

    df = (
        df.dropna(subset=[warmup_col])
        .reset_index(drop=True)
    )

    print(
        f"  Dropped {before - len(df):,} warm-up rows "
        f"without a full {LONG_ROLLING_WINDOW_HOURS}h lookback"
    )

    return df


# ============================================================================
# 3. INSERT NEW ROWS
# ============================================================================

def select_new_rows(
    engineered_df: pd.DataFrame,
    latest_stored_dt,
) -> pd.DataFrame:

    if latest_stored_dt is None:
        return engineered_df

    if getattr(latest_stored_dt, "tzinfo", None) is not None:
        latest_stored_dt = latest_stored_dt.tz_localize(None)

    dt_col = engineered_df["datetime"]

    if hasattr(dt_col.dt, "tz") and dt_col.dt.tz is not None:
        dt_col = dt_col.dt.tz_localize(None)

    return (
        engineered_df[dt_col > latest_stored_dt]
        .reset_index(drop=True)
    )


def align_dtypes_to_feature_group(
    df: pd.DataFrame,
    fg,
) -> pd.DataFrame:
    """Casts numeric columns to match the Feature Group's schema exactly."""

    schema = fg.schema

    for schema_field in schema:
        col_name = schema_field.name

        if col_name not in df.columns:
            continue

        fg_type = schema_field.type.lower()

        if "int" in fg_type:
            df[col_name] = (
                pd.to_numeric(
                    df[col_name],
                    errors="coerce",
                )
                .fillna(0)
                .astype("int64")
            )

        elif "float" in fg_type or "double" in fg_type:
            df[col_name] = (
                pd.to_numeric(
                    df[col_name],
                    errors="coerce",
                )
                .astype("float64")
            )

    return df


# ============================================================================
# 4. PRUNE OLD DATA (FIXED FOR ACTIVE FEATURE VIEWS)
# ============================================================================

def prune_old_records(
    fg,
    local_csv_path: str,
    retention_years: int,
):
    """
    Finds and deletes data older than the retention threshold.

    Uses targeted key deletion rather than table clear/overwrite to
    comply with downstream Hopsworks Feature Views.
    """

    days_back = retention_years * 365

    cutoff_date = (
        datetime.utcnow()
        - timedelta(days=days_back)
    ).replace(
        minute=0,
        second=0,
        microsecond=0,
    )

    cutoff_str = cutoff_date.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    print(
        f"Pruning rows older than {cutoff_str} "
        f"(keeping latest {retention_years} years)..."
    )

    # ------------------------------------------------------------------------
    # Step A: Prune Remote Feature Store
    # ------------------------------------------------------------------------

    try:
        # Filter for rows that have fallen out-of-bounds
        old_df = (
            fg.filter(
                fg.datetime < cutoff_str
            )
            .read()
        )

        if old_df is not None and not old_df.empty:

            print(
                f"  Found {len(old_df):,} old rows "
                "to prune from Hopsworks Feature Store."
            )

            # Hopsworks delete requires a DataFrame
            # containing at least the primary key column.
            deletion_keys = pd.DataFrame(
                {
                    "datetime": old_df["datetime"]
                }
            )

            # Execute targeted row eviction
            fg.delete(deletion_keys)

            print(
                "  Hopsworks row pruning complete."
            )

        else:
            print(
                "  No out-of-bounds rows found in Hopsworks."
            )

    except Exception as e:
        print(
            f"  WARNING: Hopsworks key-based pruning failed: {e}"
        )

    # ------------------------------------------------------------------------
    # Step B: Prune Local CSV Mirror
    # ------------------------------------------------------------------------

    if os.path.exists(local_csv_path):

        try:
            local_df = pd.read_csv(local_csv_path)

            local_df["datetime"] = pd.to_datetime(
                local_df["datetime"]
            )

            # Filter to keep only rows newer than
            # or equal to the cutoff.
            before_len = len(local_df)

            filtered_df = local_df[
                local_df["datetime"] >= cutoff_date
            ].copy()

            after_len = len(filtered_df)

            # Format dates back to string format to
            # maintain CSV style consistency.
            filtered_df["datetime"] = (
                filtered_df["datetime"]
                .dt.strftime("%Y-%m-%d %H:%M:%S")
            )

            filtered_df.to_csv(
                local_csv_path,
                index=False,
            )

            print(
                f"  Updated local CSV: "
                f"{before_len:,} -> {after_len:,} rows "
                f"({before_len - after_len:,} pruned)."
            )

        except Exception as e:
            print(
                f"  WARNING: Local CSV pruning failed: {e}"
            )


# ============================================================================
# MAIN EXECUTION ENTRYPOINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--city",
        type=str,
        required=True,
        help="Target city profile name",
    )

    parser.add_argument(
        "--skip-upwind",
        action="store_true",
        help="Skip upwind calculation logic",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------------
    # Initialize connection to Feature Store
    # ------------------------------------------------------------------------

    fs = get_hopsworks_feature_store()

    fg = get_feature_group(
        fs,
        args.city,
    )

    latest_stored_dt = get_latest_stored_datetime(
        fs,
        args.city,
    )

    # ------------------------------------------------------------------------
    # Calculate scheduling window based on history tail
    # ------------------------------------------------------------------------

    if latest_stored_dt is None:
        # Default cold start
        start_date = (
            datetime.utcnow()
            - timedelta(days=LOOKBACK_BUFFER_DAYS)
        ).strftime("%Y-%m-%d")

    else:
        start_date = (
            latest_stored_dt
            - timedelta(days=LOOKBACK_BUFFER_DAYS)
        ).strftime("%Y-%m-%d")

    end_date = datetime.utcnow().strftime(
        "%Y-%m-%d"
    )

    # ------------------------------------------------------------------------
    # 2. Extract and Process the Lookback Window
    # ------------------------------------------------------------------------

    lat, lon = geocode_city(args.city)

    raw_df, upwind_cols = fetch_recent_window(
        lat,
        lon,
        start_date,
        end_date,
        args.skip_upwind,
    )

    engineered_df = engineer_features(
        raw_df,
        upwind_cols,
        args.skip_upwind,
    )

    # ------------------------------------------------------------------------
    # 3. Isolate New Data Points
    # ------------------------------------------------------------------------

    new_rows = select_new_rows(
        engineered_df,
        latest_stored_dt,
    )

    # ------------------------------------------------------------------------
    # 4. Insert New Items if Timeline Has Progressed
    # ------------------------------------------------------------------------

    if not new_rows.empty:

        print(
            f"Inserting {len(new_rows):,} incremental "
            "records into Feature Group..."
        )

        new_rows = align_dtypes_to_feature_group(
            new_rows,
            fg,
        )

        fg.insert(
            new_rows,
            write_options={
                "wait_for_job": True
            },
        )

        # Append to the local CSV mirror tracking file
        local_csv = (
            f"aqi_features_{args.city.lower()}.csv"
            if not os.path.exists("aqi_features.csv")
            else "aqi_features.csv"
        )

        if os.path.exists(local_csv):

            try:
                historical_csv_df = pd.read_csv(
                    local_csv
                )

                combined_df = (
                    pd.concat(
                        [
                            historical_csv_df,
                            new_rows,
                        ]
                    )
                    .drop_duplicates(
                        subset=["datetime"]
                    )
                )

                combined_df.to_csv(
                    local_csv,
                    index=False,
                )

            except Exception as e:

                print(
                    f"  Failed local mirror update: {e}. "
                    "Writing new rows straight to file."
                )

                new_rows.to_csv(
                    local_csv,
                    index=False,
                )

        else:
            new_rows.to_csv(
                local_csv,
                index=False,
            )

    else:

        print(
            "No new rows to insert "
            "(source data hasn't advanced past "
            "the latest stored hour yet)."
        )

    # ------------------------------------------------------------------------
    # 5. Clean Out Trailing Historical History Safely
    # ------------------------------------------------------------------------

    local_csv_target = (
        "aqi_features.csv"
        if os.path.exists("aqi_features.csv")
        else f"aqi_features_{args.city.lower()}.csv"
    )

    prune_old_records(
        fg,
        local_csv_target,
        RETENTION_YEARS,
    )

    print("Hourly update complete.")


if __name__ == "__main__":
    main()
