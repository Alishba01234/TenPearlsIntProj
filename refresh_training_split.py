"""
refresh_training_split.py
Run once daily, BEFORE train_models.py, as part of the daily retraining
job.

Why this script exists: aqiPipeline.py's create_feature_view_split()
registers a Feature View training-dataset split (fv.create_train_test_split)
that is a materialized SNAPSHOT, not a live query -- train_models.py reads
it by a fixed --td-version (default 1). hourly_feature_update.py keeps
appending new rows to the underlying feature GROUP, but that snapshot
never automatically updates. Without this refresh step, "daily retraining"
would silently retrain on the exact same frozen data every day and never
actually see the new hourly rows.

This script re-reads the CURRENT state of the feature group, recomputes
split boundaries (train = everything but the most recent 12 months, test =
the most recent 12 months, same logic as aqiPipeline.py), and creates a
NEW training-dataset version under the existing Feature View. It writes
that new version number to latest_td_version.txt so the training workflow
can pass it to train_models.py --td-version.

Retention note: the underlying feature group is NOT physically pruned --
Hudi row-level deletes need the Spark engine, which the hourly pipeline
doesn't have (see hourly_feature_update.py's retention_filter() for the
full explanation), so it grows forever. Reading it here with fg.read()
unfiltered would mean pulling and materializing a little more history
every single day, forever, only for train_models.py's own retention
filter to throw most of it away downstream. Instead, this script bounds
the read itself to --retention-years, so both the read and the resulting
Training Dataset snapshot stay a fixed, bounded size. train_models.py
still applies its own retention filter on top of whatever it's handed --
that's cheap defense-in-depth for older training-dataset versions created
before this filtering existed, or created with a different
--retention-years value -- not something this script's filtering makes
redundant.

Read-only with respect to aqiPipeline.py / train_models.py -- only imports
from aqiPipeline.py.

Usage:
    python refresh_training_split.py --city Karachi
    # then:
    python train_models.py --city karachi --td-version $(cat latest_td_version.txt)
"""
import argparse

import pandas as pd

from aqiPipeline import (  # noqa: F401 -- read-only reuse
    get_hopsworks_feature_store, prepare_for_split, compute_split_boundaries,
    create_feature_view_split, TARGET_COLS,
)

VERSION_FILE = "latest_td_version.txt"

# Keep in sync with RETENTION_YEARS in hourly_feature_update.py /
# train_models.py.
RETENTION_YEARS = 4


def retention_cutoff_str(retention_years: int = RETENTION_YEARS) -> str:
    """The same read-time retention mechanism used elsewhere in this
    pipeline (see hourly_feature_update.py's retention_filter()) -- bounds
    what gets read from the feature group to the last `retention_years`
    years, since the feature group itself is never physically pruned."""
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=retention_years)
    return cutoff.strftime("%Y-%m-%d %H:%M:%S")


def get_new_training_dataset_version(fv, fallback_hint: int = None):
    """Best-effort lookup of the version number that create_train_test_split()
    just created. The exact hsfs API for reading this back can differ
    slightly across hopsworks/hsfs versions, so this tries a couple of
    approaches and falls back to a locally tracked counter if neither
    works -- verify against the Hopsworks UI (Feature View -> Training
    Datasets tab) if this ever looks wrong."""
    try:
        tds = fv.get_training_datasets()
        versions = [td.version for td in tds]
        if versions:
            return max(versions)
    except Exception as e:
        print(f"  Could not list training datasets via get_training_datasets() ({e})")

    if fallback_hint is not None:
        print(f"  Falling back to locally tracked counter: v{fallback_hint}")
        return fallback_hint

    print("  WARNING: could not determine the new training dataset version. "
          "Check the Hopsworks UI (Feature View -> Training Datasets) and pass "
          "--td-version manually to train_models.py for this run.")
    return None


def read_local_counter() -> int:
    try:
        with open(VERSION_FILE) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="Karachi")
    parser.add_argument("--fg-version", type=int, default=1)
    parser.add_argument("--fv-version", type=int, default=1)
    parser.add_argument("--retention-years", type=int, default=RETENTION_YEARS,
                         help="Only read/split rows within this many years of now, even though "
                              "the feature group itself holds more (default: %(default)s, "
                              "keep in sync with the other pipeline scripts)")
    args = parser.parse_args()

    city_key = args.city.lower().replace(" ", "_")

    fs = get_hopsworks_feature_store()
    fg = fs.get_feature_group(f"aqi_features_{city_key}", version=args.fg_version)

    print("Reading current feature group state...")
    cutoff_str = retention_cutoff_str(args.retention_years)
    print(f"  Bounding read to rows on/after {cutoff_str} "
          f"(retention window: {args.retention_years} years) -- the feature "
          f"group itself may hold more, see module docstring")
    df = fg.filter(fg.datetime >= cutoff_str).read()
    print(f"  {len(df):,} rows within the retention window")

    df_for_split = prepare_for_split(df)
    bounds = compute_split_boundaries(df_for_split)

    print("\nCreating a fresh training-dataset split (new version) under the existing Feature View...")
    fv = create_feature_view_split(fs, df_for_split, city_key, bounds, version=args.fv_version)

    local_counter = read_local_counter()
    new_version = get_new_training_dataset_version(fv, fallback_hint=local_counter + 1)

    if new_version is not None:
        with open(VERSION_FILE, "w") as f:
            f.write(str(new_version))
        print(f"\nWrote {VERSION_FILE} = {new_version}")
    else:
        print(f"\nCould NOT write {VERSION_FILE} -- see warning above.")


if __name__ == "__main__":
    main()
