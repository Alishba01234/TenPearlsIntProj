"""
shap_utils.py
SHAP-based explainability for whichever model won each horizon (Ridge,
Random Forest, XGBoost, or the Keras neural net). Picks the right SHAP
explainer per model type since they don't share one API, and falls back to
a model-agnostic KernelExplainer for anything else (e.g. the neural net).
"""
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from train_models import KerasWrapper  # read-only import, same as model_service.py


def _build_explainer(model, background: np.ndarray):
    if isinstance(model, (RandomForestRegressor, XGBRegressor)):
        return shap.TreeExplainer(model)
    if isinstance(model, Ridge):
        return shap.LinearExplainer(model, background)
    # KerasWrapper or anything else: generic, model-agnostic fallback.
    sample_size = min(50, len(background))
    idx = np.random.choice(len(background), size=sample_size, replace=False)
    return shap.KernelExplainer(model.predict, background[idx])


def explain_prediction(model, scaler, feature_columns: list, background_df: pd.DataFrame,
                        instance_row: pd.Series, top_n: int = 10) -> dict:
    """Local explanation for one prediction (the current 24h/48h/72h
    forecast) plus a global feature-importance ranking computed over a
    background sample of recent historical rows."""
    X_bg = scaler.transform(background_df[feature_columns])
    X_instance = scaler.transform(
        pd.DataFrame([instance_row[feature_columns].values], columns=feature_columns)
    )

    explainer = _build_explainer(model, X_bg)
    shap_values_instance = explainer.shap_values(X_instance)
    shap_values_bg = explainer.shap_values(X_bg[: min(100, len(X_bg))])

    # Some explainers return a list (one array per output) even for a
    # single-output regressor -- normalize to a plain array either way.
    if isinstance(shap_values_instance, list):
        shap_values_instance = shap_values_instance[0]
    if isinstance(shap_values_bg, list):
        shap_values_bg = shap_values_bg[0]

    local_vals = np.array(shap_values_instance).reshape(-1)
    local_ranked = sorted(zip(feature_columns, local_vals), key=lambda t: abs(t[1]), reverse=True)[:top_n]

    global_importance = np.abs(np.array(shap_values_bg)).mean(axis=0)
    global_ranked = sorted(zip(feature_columns, global_importance), key=lambda t: t[1], reverse=True)[:top_n]

    return {
        "local": [{"feature": f, "shap_value": round(float(v), 3)} for f, v in local_ranked],
        "global": [{"feature": f, "mean_abs_shap": round(float(v), 3)} for f, v in global_ranked],
    }
