"""
Run once daily, BEFORE train_models.py, as part of the daily retraining
job. Why this script exists: aqiPipeline.py's create_feature_view_split()
registers a Feature View training-dataset split (fv.create_train_test_split)
that is a materialized SNAPSHOT, not a live query -- train_models.py reads
it by a fixed --td-version (default 1). hourly_feature_update.py keeps
appending new rows to the underlying feature GROUP, but that snapshot
never automatically updates. Without this refresh step, "daily retraining"
would silently retrain on the exact same frozen data every day and never
actually see the new hourly rows.

This script re-reads the CURRENT state of the feature group, recomputes
split boundaries (train = everything but the most recent 12 months, test =
the most recent 12 months, same logic as aqiPipeline.py), and registers a
NEW training-dataset version under the Feature View via
create_feature_view_split(), which keeps the most recent `keep_last_splits`
versions (default 7) and deletes older ones -- see aqiPipeline.py for why
this replaced the earlier "always delete everything and land back at v1"
approach: reusing version 1 let train_models.py's read, running seconds
later, race Hopsworks' own read-path caching for that reused version
number and occasionally see a stale/partial result (this was traced back
as the likely root cause of an intermittent "Input contains NaN" crash in
train_models.py). Always creating a new version sidesteps that race
entirely.

Hopsworks assigns the training-dataset version number itself, so this
script reads back whatever version actually got created (always the max
version on the Feature View) and writes it to latest_td_version.txt, so
train_models.py --td-version always points at something that genuinely
exists. Before trusting that version number, it also does a best-effort
read-back verification (a few retries with backoff) to catch the case
where the version was *just* created and the read path hasn't caught up
yet -- belt-and-suspenders on top of the versioning fix above, not a
replacement for it.

Read-only with respect to aqiPipeline.py / train_models.py -- only imports
from aqiPipeline.py.

Usage:
    python refresh_training_split.py --city Karachi
    python train_models.py --city karachi --td-version $(cat latest_td_version.txt)
"""
import argparse
import time

from aqiPipeline import (  # noqa: F401 -- read-only reuse
    get_hopsworks_feature_store, prepare_for_split, compute_split_boundaries,
    create_feature_view_split, TARGET_COLS,
)
from hopsworks_read_utils import robust_read, robust_train_test_split

VERSION_FILE = "latest_td_version.txt"
KEEP_LAST_SPLITS = 7  # must match (or be <=) aqiPipeline.py's create_feature_view_split default
VERIFY_ATTEMPTS = 3
VERIFY_ROW_TOLERANCE = 5  # small slack for edge-row filtering differences between runs


def get_new_training_dataset_version(fv, fallback_hint: int = None):
    """Best-effort lookup of the version number that create_train_test_split()
    just created. The exact hsfs API for reading this back can differ
    slightly across hopsworks/hsfs versions, so this tries the direct
    lookup first and falls back to fallback_hint if that lookup fails --
    verify against the Hopsworks UI (Feature View -> Training Datasets tab)
    if this ever looks wrong."""
    if fv is not None:
        try:
            tds = fv.get_training_datasets()
            versions = [td.version for td in tds]
            if versions:
                return max(versions)
        except Exception as e:
            print(f"  Could not list training datasets via get_training_datasets() ({e})")

    if fallback_hint is not None:
        print(f"  Falling back to hint: v{fallback_hint}")
        return fallback_hint

    print("  WARNING: could not determine the new training dataset version. "
          "Check the Hopsworks UI (Feature View -> Training Datasets) and pass "
          "--td-version manually to train_models.py for this run.")
    return None


def verify_split_readable(fv, version: int, expected_test_rows: int) -> bool:
    """Reads the newly created split back (with retries) and checks the
    test-row count roughly matches what we just asked Hopsworks to create,
    before letting train_models.py proceed against it. This does not
    replace the versioning fix in create_feature_view_split() -- it's a
    second line of defense in case a genuinely new version is still briefly
    stale on the read path right after creation."""
    for attempt in range(1, VERIFY_ATTEMPTS + 1):
        try:
            _, X_test_check, _, _ = robust_train_test_split(
                fv, training_dataset_version=version, label="refresh_verify"
            )
            n = len(X_test_check)
            if abs(n - expected_test_rows) <= VERIFY_ROW_TOLERANCE:
                print(f"  Verified: read back {n} test rows for v{version} "
                      f"(expected ~{expected_test_rows}). OK.")
                return True
            print(f"  Attempt {attempt}/{VERIFY_ATTEMPTS}: read back {n} test rows for v{version}, "
                  f"expected ~{expected_test_rows} -- looks stale, retrying...")
        except Exception as e:
            print(f"  Attempt {attempt}/{VERIFY_ATTEMPTS}: verification read failed ({e}), retrying...")
        if attempt < VERIFY_ATTEMPTS:
            time.sleep(20 * attempt)
    return False


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

    # prepare_for_split() only drops trailing-edge rows lacking full target
    # history yet (rows too new to have 72h of future data). It does NOT
    # scan for NaNs elsewhere in the timeline -- e.g. from a missed hourly
    # run, or a genuine gap in the upstream Open-Meteo data -- so do that
    # explicitly here rather than assuming the edge drop already caught it.
    check_cols = ["us_aqi"] + TARGET_COLS
    na_mask = df_for_split[check_cols].isna().any(axis=1)
    n_bad = int(na_mask.sum())
    if n_bad:
        print(f"  Dropping {n_bad} row(s) with NaN in us_aqi/target columns "
              f"(not caught by prepare_for_split's edge-only filter):")
        print(df_for_split.loc[na_mask, ["datetime"]].to_string(index=False))
        df_for_split = df_for_split[~na_mask].reset_index(drop=True)

    bounds = compute_split_boundaries(df_for_split)
    expected_test_rows = int((df_for_split["datetime"] >= bounds["test_start"]).sum())
    print(f"\nCreating a new training-dataset split under the existing Feature "
          f"View (older versions are kept, up to the last {KEEP_LAST_SPLITS}, "
          f"not all deleted -- see aqiPipeline.py for why this replaced the "
          f"old always-v1 approach)...")
    try:
        fv = create_feature_view_split(
            fs, df_for_split, city_key, bounds, version=args.fv_version, keep_last_splits=KEEP_LAST_SPLITS
        )
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

    new_version = get_new_training_dataset_version(fv, fallback_hint=None)
    if new_version is None:
        print(f"\nCould NOT write {VERSION_FILE} -- see warning above.")
        return

    if fv is not None:
        verified = verify_split_readable(fv, new_version, expected_test_rows)
        if not verified:
            raise RuntimeError(
                f"Could not verify the new split (v{new_version}) is readable with the "
                f"expected row count (~{expected_test_rows} test rows) after "
                f"{VERIFY_ATTEMPTS} attempts -- refusing to point train_models.py at what "
                f"may be stale/partial data. Check the Hopsworks UI (Feature View -> "
                f"Training Datasets) for v{new_version} before re-running."
            )
    else:
        print("  Skipping read-back verification -- Feature View handle unavailable.")

    with open(VERSION_FILE, "w") as f:
        f.write(str(new_version))
    print(f"\nWrote {VERSION_FILE} = {new_version}")


if __name__ == "__main__":
    main()
