"""
monthly_prune.py
Monthly retention job -- companion to hourly_feature_update.py.

hourly_feature_update.py now only appends new rows, every hour. This
script does the retention side separately, once a month: it deletes
whichever calendar month(s) of data have aged out of the
--retention-years window, from both the Hopsworks Feature Store and the
committed aqi_features.csv.

Why split out of the hourly script: running a Hopsworks read+delete pass
every hour meant 24 API round-trips a day for what is, almost every hour,
zero rows to prune -- the window only actually advances into new stale
rows once a month. Pruning here instead, once a month, cuts that down to
12 API round-trips a year with no change to what data survives.

CUTOFF LOGIC -- calendar-month anchored, not "now - N years":
    The cutoff is pinned to the 1st of the CURRENT month, then stepped
    back --retention-years years. E.g. running this on any day in
    September 2026 with --retention-years 4 gives a cutoff of
    2026-09-01 minus 4 years = 2022-09-01 00:00:00 -- rows strictly
    before that are dropped, rows from 2022-09-01 onward are kept.
    Anchoring to the 1st of the month (rather than the exact run
    timestamp) makes this idempotent: re-running mid-month, or the job
    firing a few hours late, produces the same cutoff and therefore the
    same result.

    Note this is EXACTLY 4 years back, not 4 years + change. If you want
    to always keep at least one buffer month on top of the retention
    window, subtract an extra month when computing `cutoff` below.

Meant to be run once a month by GitHub Actions -- see
.github/workflows/monthly_prune.yml

Usage:
    python monthly_prune.py --city "Karachi"
"""
import argparse
import os
from datetime import datetime

import pandas as pd

from aqiPipeline import get_hopsworks_feature_store, delete_existing_csv
from hourly_feature_update import get_feature_group, FEATURE_GROUP_VERSION

RETENTION_YEARS = 4


# ==================================================================
# 1. COMPUTE THE CUTOFF
# ==================================================================

def get_monthly_retention_cutoff(retention_years: int = RETENTION_YEARS, as_of: datetime = None) -> pd.Timestamp:
    """First-of-month cutoff, `retention_years` back from the first of the
    month containing `as_of` (defaults to now). Rows strictly before this
    timestamp are out of the retention window."""
    as_of = as_of or datetime.utcnow()
    first_of_this_month = pd.Timestamp(year=as_of.year, month=as_of.month, day=1)
    return first_of_this_month - pd.DateOffset(years=retention_years)


# ==================================================================
# 2. PRUNE HOPSWORKS
# ==================================================================

def prune_feature_group(fg, cutoff: pd.Timestamp):
    """Deletes rows older than `cutoff` from this Hudi-backed feature group
    (aqiPipeline.py creates it with time_travel_format="HUDI") via HSFS's
    commit_delete_record(), which needs a DataFrame of the rows to remove
    (matched by primary key + event time)."""
    print(f"Pruning Hopsworks rows older than {cutoff} ...")
    try:
        old_df = fg.filter(fg.datetime < cutoff.strftime("%Y-%m-%d %H:%M:%S")).read()
    except Exception as e:
        print(f"  Could not read old rows for pruning, aborting this run's prune step: {e}")
        return

    if old_df is None or old_df.empty:
        print("  No rows older than the retention window -- nothing to prune.")
        return

    old_df["datetime"] = pd.to_datetime(old_df["datetime"])
    print(f"  Deleting {len(old_df):,} rows ({old_df['datetime'].min()} -> {old_df['datetime'].max()})...")
    try:
        fg.commit_delete_record(old_df)
        print("  Pruned.")
    except Exception as e:
        print(f"  WARNING: commit_delete_record failed ({e}). If your installed hopsworks/hsfs "
              f"version exposes a different row-delete API, this call needs updating -- "
              f"check the Hopsworks docs for your version's Feature Group delete method.")


# ==================================================================
# 3. PRUNE THE LOCAL CSV
# ==================================================================

def prune_local_csv(csv_path: str, cutoff: pd.Timestamp):
    if not os.path.exists(csv_path):
        print(f"  {csv_path} not found -- nothing to prune locally.")
        return

    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    before = len(df)
    df = df[df["datetime"] >= cutoff].sort_values("datetime").reset_index(drop=True)
    n_dropped = before - len(df)

    if n_dropped == 0:
        print(f"  {csv_path}: no rows older than the retention window -- nothing to prune.")
        return

    delete_existing_csv(csv_path)
    df.to_csv(csv_path, index=False)
    print(f"  Updated {csv_path}: {before:,} -> {len(df):,} rows "
          f"(dropped {n_dropped:,} rows older than {cutoff.date()})")


# ==================================================================
# MAIN
# ==================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="Karachi")
    parser.add_argument("--out", default="aqi_features.csv")
    parser.add_argument("--retention-years", type=int, default=RETENTION_YEARS)
    args = parser.parse_args()

    city_key = args.city.lower().replace(" ", "_")
    cutoff = get_monthly_retention_cutoff(args.retention_years)
    print(f"Retention window: latest {args.retention_years} years -- cutoff for this run is {cutoff}")

    fs = get_hopsworks_feature_store()
    fg = get_feature_group(fs, city_key, version=FEATURE_GROUP_VERSION)

    prune_feature_group(fg, cutoff)
    prune_local_csv(args.out, cutoff)

    print("\nMonthly prune complete.")


if __name__ == "__main__":
    main()
