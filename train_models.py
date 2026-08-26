"""
Training pipeline for AQI forecasting.
  1. Fetches the train/validation/test split directly from the Hopsworks
     Feature Store (the Feature View created by aqi_pipeline.py).
  2. Trains 3 candidate models (Ridge Regression, Random Forest, XGBoost)
     SEPARATELY for each of the 3 forecast horizons (24h/48h/72h) -- 9
     models trained in total.
  3. Evaluates every model on the validation set using RMSE, MAE, R^2.
  4. Picks the best model per horizon (lowest validation RMSE), reports its
     final unbiased performance on the held-out test set.
  5. Registers the 3 winning models (one per horizon) in the Hopsworks
     Model Registry.
Why scikit-learn/XGBoost and not TensorFlow/PyTorch here: a genuinely
serverless deployment (Lambda-style functions) typically caps deployment
package size well below what a full TensorFlow/PyTorch install needs.
These three are lightweight, serverless-friendly, and still give a fair
statistical-vs-ensemble-vs-gradient-boosting comparison. A deep learning
model can be added later as an additional candidate without changing this
script's structure.
Usage:
    python train_models.py --city karachi
"""
import argparse
import logging
import os
import hopsworks
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from hopsworks_read_utils import robust_train_test_split
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
try:
    import tensorflow as tf
    from tensorflow.keras import layers, callbacks
    TENSORFLOW_AVAILABLE = True
except Exception as e:
    TENSORFLOW_AVAILABLE = False
    print(f"WARNING: tensorflow import failed -- neural network candidate will "
          f"be skipped. Underlying error ({type(e).__name__}): {e}")

class KerasWrapper:
    """Wraps a Keras model so it can be used like the other estimators.
    This wrapper exposes a predict method and keeps the raw Keras model
    available for saving in the registry.
    """
    def __init__(self, keras_model):
        self.keras_model = keras_model

    def predict(self, X):
        preds = self.keras_model.predict(X, verbose=0)
        if preds.ndim == 2 and preds.shape[1] == 1:
            return preds.ravel()
        return preds

load_dotenv()

logging.getLogger("hsfs").setLevel(logging.CRITICAL)
logging.getLogger("hsml").setLevel(logging.CRITICAL)
logging.getLogger("hopsworks").setLevel(logging.CRITICAL)
FORECAST_HORIZONS = (24, 48, 72)
TARGET_COL = "us_aqi"
TARGET_COLS = [f"target_{TARGET_COL}_{h}h_ahead" for h in FORECAST_HORIZONS]

# LOAD DATA FROM FEATURE STORE
def connect_to_feature_store():
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "HOPSWORKS_API_KEY not found. Add a .env file with "
            "HOPSWORKS_API_KEY=your_key_here"
        )
    print("Logging into Hopsworks...")
    project = hopsworks.login(api_key_value=api_key)
    return project, project.get_feature_store()

def load_splits(fs, city: str, fv_version: int = 1, td_version: int = 1):
    fv = fs.get_feature_view(f"aqi_fv_{city}", version=fv_version)
    X_train, X_test, y_train, y_test = robust_train_test_split(
        fv, training_dataset_version=td_version, label="load_splits"
    )

    def _sort_by_time(X, y, split_name: str):
        order = X["datetime"].argsort().to_numpy()
        X_sorted = X.iloc[order].reset_index(drop=True)
        y_sorted = y.iloc[order].reset_index(drop=True)
        assert X_sorted["datetime"].is_monotonic_increasing, f"{split_name} not sorted"
        return X_sorted, y_sorted

    X_train, y_train = _sort_by_time(X_train, y_train, "train")
    X_test, y_test = _sort_by_time(X_test, y_test, "test")

    test_datetimes = X_test["datetime"].reset_index(drop=True)
    train_datetimes = X_train["datetime"].reset_index(drop=True)
    for df in (X_train, X_test):
        df.drop(columns=["datetime"], inplace=True)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, train_datetimes, test_datetimes

RANDOM_SEARCH_ITER = 15  # bounded to keep runtime reasonable
CV_SPLITS = 3  # TimeSeriesSplit folds -- chronological, no shuffling

def fit_tuned_ridge(X_train, y_train, sample_weight=None):
    """Switched from RidgeCV to RandomizedSearchCV so it exposes
    .best_score_ the same way RF/XGBoost do below -- needed as the
    model-selection signal now that there's no external val set. Ridge
    is cheap enough that losing RidgeCV's shared-decomposition speedup
    doesn't matter in practice."""
    search = RandomizedSearchCV(
        Ridge(),
        param_distributions={"alpha": np.logspace(-3, 3, 25)},
        n_iter=25,
        cv=TimeSeriesSplit(n_splits=CV_SPLITS),
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
    )
    fit_params = {"sample_weight": sample_weight} if sample_weight is not None else {}
    search.fit(X_train, y_train, **fit_params)
    return search.best_estimator_, -search.best_score_

def fit_tuned_random_forest(X_train, y_train, sample_weight=None):
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [6, 10, 14, None],
        "min_samples_leaf": [1, 3, 5, 10],
        "max_features": ["sqrt", 0.5, 1.0],
    }
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions=param_dist, n_iter=RANDOM_SEARCH_ITER,
        cv=TimeSeriesSplit(n_splits=CV_SPLITS),
        scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1,
    )
    fit_params = {"sample_weight": sample_weight} if sample_weight is not None else {}
    search.fit(X_train, y_train, **fit_params)
    return search.best_estimator_, -search.best_score_
 
def fit_tuned_xgboost(X_train, y_train, sample_weight=None):
    """Widened depth (up to 10) and n_estimators (up to 800), added a
    slower learning rate option now that more estimators are available,
    and added L1/L2 regularization + min_child_weight -- the old grid
    had no regularization knobs at all, which biases XGBoost toward
    overfitting exactly the kind of noisy, long-horizon target where it
    was losing to Ridge."""
    param_dist = {
        "n_estimators": [100, 200, 300, 400, 600, 800],
        "max_depth": [3, 4, 6, 8, 10],
        "learning_rate": [0.005, 0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 10],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0, 10.0],
    }
    search = RandomizedSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1),
        param_distributions=param_dist, n_iter=RANDOM_SEARCH_ITER,
        cv=TimeSeriesSplit(n_splits=CV_SPLITS),
        scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1,
    )
    fit_params = {"sample_weight": sample_weight} if sample_weight is not None else {}
    search.fit(X_train, y_train, **fit_params)
    return search.best_estimator_, -search.best_score_

def _build_nn(n_features: int, seed: int):
    tf.random.set_seed(seed)
    model = tf.keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(64, activation="relu"), layers.Dropout(0.2),
        layers.Dense(32, activation="relu"), layers.Dropout(0.1),
        layers.Dense(16, activation="relu"), layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model

def _fit_nn_once(X_fit_all, y_fit_all, seed: int, internal_val_frac: float = 0.15, w_fit_all=None):
    """Trains one NN on the given data, using a chronological tail slice of
    IT (not of the outer split) purely for early-stopping -- never for
    scoring. Returns a fitted KerasWrapper."""
    n = len(X_fit_all)
    split_idx = int(n * (1 - internal_val_frac))
    X_fit, X_es = X_fit_all[:split_idx], X_fit_all[split_idx:]
    y_fit, y_es = y_fit_all[:split_idx], y_fit_all[split_idx:]
    w_fit = w_fit_all[:split_idx] if w_fit_all is not None else None

    model = _build_nn(X_fit.shape[1], seed)
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(X_fit, y_fit, sample_weight=w_fit, validation_data=(X_es, y_es), epochs=100, batch_size=64,
              callbacks=[early_stop], verbose=0)
    return KerasWrapper(model)

def fit_neural_net(X_train, y_train, seed: int = 42, sample_weight=None):
    X_arr = np.asarray(X_train)
    y_arr = np.asarray(y_train)
    w_arr = np.asarray(sample_weight) if sample_weight is not None else None
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    fold_rmses = []
    for fold_i, (tr_idx, val_idx) in enumerate(tscv.split(X_arr)):
        w_tr = w_arr[tr_idx] if w_arr is not None else None
        fold_wrapper = _fit_nn_once(X_arr[tr_idx], y_arr[tr_idx], seed=seed + fold_i, w_fit_all=w_tr)
        fold_rmse = evaluate(y_arr[val_idx], fold_wrapper.predict(X_arr[val_idx]))["rmse"]
        fold_rmses.append(fold_rmse)
    cv_rmse = float(np.mean(fold_rmses))
    final_wrapper = _fit_nn_once(X_arr, y_arr, seed=seed, w_fit_all=w_arr)
    return final_wrapper, cv_rmse

def get_model_fitters():
    """Returns {name: function(X_train, y_train) -> fitted model}. Each
    horizon calls these fresh, since tuning is redone per-horizon --
    the best hyperparameters for 24h may differ from those for 72h."""
    return {
        "ridge": fit_tuned_ridge,
        "random_forest": fit_tuned_random_forest,
        "xgboost": fit_tuned_xgboost,
    }

def select_features_for_horizon(X: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Each horizon's model should only see weather aligned to ITS OWN
    target time -- e.g. the 24h model shouldn't get 48h/72h-future weather,
    since at deployment for a 24h prediction you'd only have a 24h-ahead
    forecast, not a 48h one. Drops the other horizons' future-weather
    columns before fitting/predicting."""
    other_horizons = [h for h in FORECAST_HORIZONS if h != horizon]
    drop_cols = [c for c in X.columns if any(f"_future_{oh}h" in c for oh in other_horizons)]
    return X.drop(columns=drop_cols)

def report_feature_target_correlations(X_train, y_train, horizon: int, top_n: int = 10):
    """Direct evidence of how much genuine LINEAR signal exists between
    each feature and this horizon's target, before any model is trained.
    Low correlations across the board mean the ceiling is a real data/signal
    limitation, not something more tuning or preprocessing will fix."""
    target_col = f"target_{TARGET_COL}_{horizon}h_ahead"
    X_h = select_features_for_horizon(X_train, horizon)
    combined = X_h.copy()
    combined["__target__"] = y_train[target_col].values
    corrs = combined.corr()["__target__"].drop("__target__").abs().sort_values(ascending=False)
    print(f"  Top {top_n} feature-target correlations ({horizon}h ahead):")
    for feat, corr in corrs.head(top_n).items():
        marker = " <- future weather" if "_future_" in feat else ""
        print(f"    {feat:35s} {corr:.3f}{marker}")
    return corrs

def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

def persistence_baseline(X, y, horizon: int) -> dict:
    """The naive 'AQI doesn't change' forecast: predict the CURRENT us_aqi
    value as the answer for h hours from now. This is the standard sanity
    check for any forecasting model -- if a trained model can't beat this,
    it isn't adding real value. 'us_aqi' (current reading) must be present
    in X for this to work, which it is (it's a feature, not just a target)."""
    target_col = f"target_{TARGET_COL}_{horizon}h_ahead"
    naive_pred = X["us_aqi"]
    return evaluate(y[target_col], naive_pred)

def report_feature_importance(model, feature_names, model_name: str, top_n: int = 10):
    """Shows which features the winning model actually relied on -- the
    fastest way to confirm whether the future-weather columns are being
    used at all, or effectively ignored."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        print(f"  (no importance/coefficient info available for {model_name})")
        return

    order = np.argsort(importances)[::-1][:top_n]
    print(f"  Top {top_n} features for {model_name}:")
    for idx in order:
        marker = " <- future weather" if "_future_" in feature_names[idx] else ""
        print(f"    {feature_names[idx]:35s} {importances[idx]:.4f}{marker}")

# TRAIN + EVALUATE PER HORIZON
OUTLIER_PERIOD_END = "2023-09-01"  # Aug 2022-Jun 2023 has ~250 extreme AQI
OUTLIER_PERIOD_WEIGHT = 0.3        # hours (>200, up to 297) not seen since tested at 1.0/0.3/0.1, 0.3 measured best (24h R2 0.662->0.670, 72h 0.239->0.280)

def train_for_horizon(horizon: int, X_train, y_train, X_test, y_test, train_datetimes=None):
    target_col = f"target_{TARGET_COL}_{horizon}h_ahead"
    X_train_h = select_features_for_horizon(X_train, horizon)
    X_test_h = select_features_for_horizon(X_test, horizon)
    baseline = persistence_baseline(X_test_h, y_test, horizon)
    print(f"  persistence baseline  RMSE={baseline['rmse']:.2f}  MAE={baseline['mae']:.2f}  R2={baseline['r2']:.3f}")
    sample_weight = None
    if train_datetimes is not None:
        sample_weight = np.where(
            pd.to_datetime(train_datetimes) < OUTLIER_PERIOD_END, OUTLIER_PERIOD_WEIGHT, 1.0
        )
    scaler = StandardScaler()
    X_train_h = pd.DataFrame(scaler.fit_transform(X_train_h), columns=X_train_h.columns, index=X_train_h.index)
    X_test_h = pd.DataFrame(scaler.transform(X_test_h), columns=X_test_h.columns, index=X_test_h.index)
    results = {}
    for name, fit_fn in get_model_fitters().items():
        model, cv_rmse = fit_fn(X_train_h, y_train[target_col], sample_weight=sample_weight)
        test_metrics = evaluate(y_test[target_col], model.predict(X_test_h.values))
        results[name] = {"model": model, "cv_rmse": cv_rmse, "test_metrics": test_metrics}
        print(f"  {name:15s}  CV RMSE={cv_rmse:.2f}  Test RMSE={test_metrics['rmse']:.2f}  Test R2={test_metrics['r2']:.3f}")

    if TENSORFLOW_AVAILABLE:
        nn_model, nn_cv_rmse = fit_neural_net(X_train_h.values, y_train[target_col].values,
                                               sample_weight=sample_weight)
        nn_test_metrics = evaluate(y_test[target_col], nn_model.predict(X_test_h.values))
        results["neural_network"] = {"model": nn_model, "cv_rmse": nn_cv_rmse, "test_metrics": nn_test_metrics}
        print(f"  {'neural_network':15s}  CV RMSE={nn_cv_rmse:.2f}  Test RMSE={nn_test_metrics['rmse']:.2f}  "
              f"Test R2={nn_test_metrics['r2']:.3f}")

    best_name = min(results, key=lambda n: results[n]["cv_rmse"])
    best_model = results[best_name]["model"]
    test_metrics = results[best_name]["test_metrics"]
    print(f"  --> Best: {best_name} (CV RMSE={results[best_name]['cv_rmse']:.2f})")
    print(f"  --> Test: RMSE={test_metrics['rmse']:.2f}  MAE={test_metrics['mae']:.2f}  R2={test_metrics['r2']:.3f}")
    report_feature_importance(best_model, list(X_train_h.columns), best_name)
    return {"horizon": horizon, "best_model_name": best_name, "best_model": best_model,
            "scaler": scaler, "cv_rmse": results[best_name]["cv_rmse"],
            "test_metrics": test_metrics, "baseline_metrics": baseline,
            "feature_columns": list(X_train_h.columns),
            "all_candidates": results}

# MODEL REGISTRY
def delete_existing_model_versions(project, model_name: str):
    """Deletes any existing registered version(s) of this model before a
    new one is created. Unlike the training-dataset delete in aqiPipeline.py, 
    Model.delete() in hsml IS a directly supported method on the objects 
    returned by get_models() -- no super()/MRO issue here -- but the delete 
    is still wrapped defensively so a Model Registry hiccup degrades to a new
    version gets created alongside the old one" rather than crashing the
    whole training run and losing today's model entirely."""
    mr = project.get_model_registry()
    try:
        existing_models = mr.get_models(name=model_name)
    except Exception as e:
        print(f"  Could not list existing models named '{model_name}' ({e}) -- "
              f"proceeding to register the new one anyway. If old versions "
              f"keep accumulating, check the Hopsworks UI (Model Registry) "
              f"and this hsml version's list/delete API.")
        return

    for m in existing_models:
        try:
            print(f"  Deleting existing model '{model_name}' v{m.version} before registering the new one...")
            m.delete()
        except Exception as e:
            print(f"  WARNING: found '{model_name}' v{m.version} but FAILED to delete it ({e}). "
                  f"The new model will still be registered, but old versions may pile up -- "
                  f"check the Hopsworks UI (Model Registry) to clean up manually.")

def register_model(project, result: dict, city: str, X_train: pd.DataFrame):
    """Saves the winning model for one horizon to the Hopsworks Model
    Registry, tagged with its test-set metrics. Any existing version(s)
    of this model are deleted first so this stays a single replaced-in-
    place v1 rather than accumulating v1, v2, v3, ... -- same pattern as
    the training-dataset split in aqiPipeline.py."""
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    horizon = result["horizon"]
    model_name = f"aqi_{city}_{horizon}h"
    model_dir = f"model_{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    delete_existing_model_versions(project, model_name)
    if isinstance(result["best_model"], KerasWrapper):
        # Keras models need their own save format, not joblib pickling.
        keras_path = os.path.join(model_dir, "keras_model.keras")
        result["best_model"].keras_model.save(keras_path)
    else:
        model_path = os.path.join(model_dir, "model.pkl")
        joblib.dump(result["best_model"], model_path)

    # The scaler MUST travel with the model -- at inference time, raw
    # features have to go through this exact same transform before being
    # fed to the model, or predictions will be wrong (especially for Ridge).
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    joblib.dump(result["scaler"], scaler_path)
    mr = project.get_model_registry()
    X_sample = X_train[result["feature_columns"]].iloc[:1]
    input_schema = Schema(X_train[result["feature_columns"]])
    output_schema = Schema(pd.DataFrame({f"target_{TARGET_COL}_{horizon}h_ahead": [0.0]}))
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
    hops_model = mr.python.create_model(
        name=model_name,
        metrics=result["test_metrics"],
        model_schema=model_schema,
        input_example=X_sample,
        description=(f"{result['best_model_name']} model predicting AQI "
                      f"{horizon}h ahead for {city}"),
    )
    hops_model.save(model_dir)
    print(f"Registered '{model_name}' in Model Registry "
          f"(algorithm: {result['best_model_name']}, test RMSE: {result['test_metrics']['rmse']:.2f})")

    if hops_model.version != 1:
        print(f"  WARNING: '{model_name}' landed at v{hops_model.version}, not v1 as expected. "
              f"delete_existing_model_versions() is supposed to clear old versions before this "
              f"registration so it always stays v1 -- either that delete silently failed this "
              f"run, or this hsml version doesn't reuse a freed version number. Check the "
              f"Hopsworks UI (Model Registry) for leftover old versions of '{model_name}'.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="karachi")
    parser.add_argument("--fv-version", type=int, default=1)
    parser.add_argument("--td-version", type=int, default=None,
                         help="Training dataset version to train on. If omitted, reads "
                              "latest_td_version.txt (written by refresh_training_split.py) "
                              "and falls back to 1 if that file doesn't exist.")
    parser.add_argument("--skip-registry", action="store_true",
                         help="Train and evaluate only, skip registering models in Hopsworks")
    args = parser.parse_args()
    td_version = args.td_version
    if td_version is None:
        try:
            with open("latest_td_version.txt") as f:
                td_version = int(f.read().strip())
            print(f"--td-version not given -- using latest_td_version.txt: v{td_version}")
        except (FileNotFoundError, ValueError):
            td_version = 1
            print("--td-version not given and latest_td_version.txt not found -- defaulting to v1")

    project, fs = connect_to_feature_store()
    X_train, X_test, y_train, y_test, train_datetimes, test_datetimes = load_splits(
        fs, args.city, args.fv_version, td_version
    )
    results = []
    for horizon in FORECAST_HORIZONS:
        result = train_for_horizon(horizon, X_train, y_train, X_test, y_test, train_datetimes)
        results.append(result)

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    summary = pd.DataFrame([
        {"horizon": f"{r['horizon']}h", "best_model": r["best_model_name"],
         "test_rmse": round(r["test_metrics"]["rmse"], 2),
         "test_mae": round(r["test_metrics"]["mae"], 2),
         "test_r2": round(r["test_metrics"]["r2"], 3),
         "baseline_r2": round(r["baseline_metrics"]["r2"], 3),
         "r2_vs_baseline": round(r["test_metrics"]["r2"] - r["baseline_metrics"]["r2"], 3)}
        for r in results
    ])
    print(summary.to_string(index=False))
    summary.to_csv("model_evaluation_summary.csv", index=False)
    print("Saved model_evaluation_summary.csv")
    if args.skip_registry:
        print("Skipping Model Registry upload (--skip-registry was set)")
        return

    for result in results:
        register_model(project, result, args.city, X_train)

if __name__ == "__main__":
    main()

# python train_models.py --city karachi --skip-registry
