"""
hopsworks_read_utils.py
Shared helper for reading from Hopsworks Feature Groups/Queries resiliently.

Why this exists: GitHub Actions runners have shown intermittent failures
talking to Hopsworks' Query Service (Arrow Flight / gRPC) -- reads hang for
minutes then fail with "Flight returned unavailable error... Socket
closed" even on modest amounts of data. This isn't a query problem, it's a
connection-stability problem specific to that transport. robust_read()
works around it by:
  1. Trying the read via the older Hive-based path (read_options=
     {"use_hive": True}), which avoids the Arrow Flight service entirely.
  2. Falling back to the default (Arrow Flight) path if the installed
     hsfs version doesn't recognize that option.
  3. Retrying either path a few times with backoff before giving up, since
     even the Hive path can hit transient network issues on a shared runner.
"""
import time


def _read_with_fallback(hive_call, default_call, retries: int = 3,
                         base_delay_seconds: int = 20, label: str = "read"):
    """Shared retry-with-backoff engine behind robust_read() and
    robust_train_test_split(): tries hive_call() first (with retries),
    then default_call() (with retries) if the hive path isn't supported
    by this hsfs version or is exhausted. Both callables take no
    arguments -- callers close over whatever params they need."""
    last_err = None

    # --- Attempt 1: Hive path (avoids the Arrow Flight Query Service) ---
    hive_supported = True
    for attempt in range(1, retries + 1):
        try:
            return hive_call()
        except TypeError:
            # This hsfs version doesn't accept read_options={"use_hive": True}
            # at all -- stop trying this path, go straight to the default one.
            hive_supported = False
            break
        except Exception as e:
            last_err = e
            print(f"  [{label}] Hive-path read failed (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(base_delay_seconds * attempt)

    if hive_supported:
        print(f"  [{label}] Hive path exhausted after {retries} attempts, "
              f"falling back to the default (Arrow Flight) path...")

    # --- Attempt 2: default path ---
    for attempt in range(1, retries + 1):
        try:
            return default_call()
        except Exception as e:
            last_err = e
            print(f"  [{label}] Default-path read failed (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(base_delay_seconds * attempt)

    raise last_err


def robust_read(readable, retries: int = 3, base_delay_seconds: int = 20, label: str = "read"):
    """readable: anything with a .read() method (a Query or FeatureGroup).
    Tries the Hive path first, then the default path, each with retries."""
    return _read_with_fallback(
        hive_call=lambda: readable.read(read_options={"use_hive": True}),
        default_call=lambda: readable.read(),
        retries=retries, base_delay_seconds=base_delay_seconds, label=label,
    )


def robust_train_test_split(fv, training_dataset_version: int, retries: int = 3,
                             base_delay_seconds: int = 20, label: str = "train_test_split"):
    """Same resilience as robust_read(), for
    FeatureView.get_train_test_split(). Split out as its own function
    (rather than reusing robust_read directly) because
    get_train_test_split() returns a 4-tuple (X_train, X_test, y_train,
    y_test), not a single DataFrame with a .read() method -- but it goes
    through the same underlying Hopsworks Query Service as fg.read(), has
    the same intermittent-failure profile on GitHub Actions runners (see
    this module's top-level docstring), and gets the same fix: Hive path
    first, default path as fallback, each retried with backoff."""
    return _read_with_fallback(
        hive_call=lambda: fv.get_train_test_split(
            training_dataset_version=training_dataset_version,
            read_options={"use_hive": True},
        ),
        default_call=lambda: fv.get_train_test_split(
            training_dataset_version=training_dataset_version,
        ),
        retries=retries, base_delay_seconds=base_delay_seconds, label=label,
    )
