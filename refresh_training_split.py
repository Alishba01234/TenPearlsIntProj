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
the most recent 12 months, same logic as aqiPipeline.py), and REPLACES the
existing training-dataset split under the Feature View: create_feature_view_split()
deletes whatever training-dataset version(s) already exist before creating
the new one, so this stays a single replaced-in-place version 1 rather
than an ever-growing v1, v2, v3, ... pile in Hopsworks storage.

Hopsworks assigns the training-dataset version number itself, and (based
on testing) reuses the freed number when nothing else remains -- but
that's not a documented API guarantee, so this script still reads back
whatever version actually got assigned and writes it to
latest_td_version.txt, and prints a clear warning if it's ever not 1, so
train_models.py --td-version always points at something that genuinely
exists even if that assumption is ever wrong on some hsfs version.

Read-only with respect to aqiPipeline.py / train_models.py -- only imports
from aqiPipeline.py.

Usage:
    python refresh_training_split.py --city Karachi
    # then:
    python train_models.py --city karachi --td-version $(cat latest_td_version.txt)
"""
import argparse

from aqiPipeline import (  # noqa: F401 -- read-only reuse
    get_hopsworks_feature_store, prepare_for_split, compute_split_boundaries,
    create_feature_view_split, TARGET_COLS,
)
from hopsworks_read_utils import robust_read

VERSION_FILE = "latest_td_version.txt"


def get_new_training_dataset_version(fv, fallback_hint: int = None):
    """Best-effort lookup of the version number that create_train_test_split()
    just created. The exact hsfs API for reading this back can differ
    slightly across hopsworks/hsfs versions, so this tries the direct
    lookup first and falls back to fallback_hint (the expected version)
    if that lookup fails -- verify against the Hopsworks UI (Feature View
    -> Training Datasets tab) if this ever looks wrong."""
    if fv is not None:
        try:
            tds = fv.get_training_datasets()
            versions = [td.version for td in tds]
            if versions:
                return max(versions)
        except Exception as e:
            print(f"  Could not list training datasets via get_training_datasets() ({e})")

    if fallback_hint is not None:
        print(f"  Falling back to expected version: v{fallback_hint}")
        return fallback_hint

    print("  WARNING: could not determine the new training dataset version. "
          "Check the Hopsworks UI (Feature View -> Training Datasets) and pass "
          "--td-version manually to train_models.py for this run.")
    return None


EXPECTED_VERSION = 1  # create_feature_view_split() deletes old training-dataset
# versions before creating a new one, so a successful run should always
# land back here -- see this module's docstring.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="Karachi")
    parser.add_argument("--fg-version", type=int, default=1)
    parser.add_argument("--fv-version", type=int, default=1)
    args = parser.parse_args()

    city_key = args.city.lower().replace(" ", "_")

    fs = get_hopsworks_feature_store()
    fg = fs.get_feature_group(f"aqi_features_{city_key}", version=args.fg_version)

    print("Reading current feature group state...")
    df = robust_read(fg, label="refresh_training_split_full_read")
    print(f"  {len(df):,} rows currently in the feature group")
    print(f"  Max datetime actually read: {df['datetime'].max()}")

    df_for_split = prepare_for_split(df)
    bounds = compute_split_boundaries(df_for_split)

    print(f"\nReplacing the training-dataset split under the existing Feature "
          f"View (old version(s) deleted first, new one should land at "
          f"v{EXPECTED_VERSION})...")
    try:
        fv = create_feature_view_split(fs, df_for_split, city_key, bounds, version=args.fv_version)
    except Exception as e:
        # The split's data can finish writing successfully even when a
        # later step (e.g. Hopsworks' post-write statistics computation)
        # throws -- see aqiPipeline.py's create_feature_view_split for the
        # specific case this guards against. Don't let that possibility
        # silently leave latest_td_version.txt pointing at yesterday's
        # split: fall back to looking the Feature View up directly and
        # still try to recover whatever version number actually landed.
        print(f"  create_feature_view_split raised ({e}); the split's data "
              f"may still have written successfully -- checking the Feature "
              f"View directly for a new version before giving up...")
        try:
            fv = fs.get_feature_view(name=f"aqi_fv_{city_key}", version=args.fv_version)
        except Exception as lookup_err:
            print(f"  Could not even look up the Feature View ({lookup_err}) -- "
                  f"this looks like a genuine failure, not just a stats-step "
                  f"hiccup after a successful write.")
            fv = None

    fallback_hint = EXPECTED_VERSION if fv is not None else None
    new_version = get_new_training_dataset_version(fv, fallback_hint=fallback_hint)

    if new_version is not None:
        if new_version != EXPECTED_VERSION:
            print(f"\n  WARNING: new training dataset landed at v{new_version}, not "
                  f"v{EXPECTED_VERSION} as expected. create_feature_view_split() is "
                  f"supposed to delete old training-dataset versions before creating "
                  f"a new one so this always stays v{EXPECTED_VERSION} -- either that "
                  f"deletion silently failed this run, or this Hopsworks/hsfs version "
                  f"doesn't reuse a freed version number the way this was designed "
                  f"around. Check the Hopsworks UI (Feature View -> Training Datasets) "
                  f"for leftover old versions. train_models.py will still be pointed "
                  f"at whatever version actually exists (v{new_version}), so training "
                  f"itself isn't broken -- this is a cleanup/versioning issue, not a "
                  f"data-correctness one.")
        with open(VERSION_FILE, "w") as f:
            f.write(str(new_version))
        print(f"\nWrote {VERSION_FILE} = {new_version}")
    else:
        print(f"\nCould NOT write {VERSION_FILE} -- see warning above.")


if __name__ == "__main__":
    main()
