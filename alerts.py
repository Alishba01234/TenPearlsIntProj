"""
alerts.py
AQI classification (US EPA scale) and hazardous-level alert generation.
"""
import math

from dashboard_config import AQI_BREAKPOINTS, ALERT_THRESHOLD, SEVERE_ALERT_THRESHOLD


def classify_aqi(value) -> dict:
    """Maps a numeric AQI value to its EPA category, color, and health message."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {"category": "Unknown", "color": "#9e9e9e", "message": "No data available.", "value": None}

    for low, high, category, color, message in AQI_BREAKPOINTS:
        if low <= value <= high:
            return {"category": category, "color": color, "message": message, "value": round(value, 1)}

    # Above the top breakpoint (>500): still classify as Hazardous rather than "Unknown".
    low, high, category, color, message = AQI_BREAKPOINTS[-1]
    return {"category": category, "color": color, "message": message, "value": round(value, 1)}


def build_alerts(predictions: dict) -> list:
    """predictions: {horizon_label: predicted_value}, e.g. {"24h": 162.3, "48h": 140.0, "72h": 205.1}
    Returns a list of alert dicts for any horizon crossing ALERT_THRESHOLD / SEVERE_ALERT_THRESHOLD."""
    alerts = []
    for horizon, value in predictions.items():
        if value is None:
            continue
        info = classify_aqi(value)
        if value >= SEVERE_ALERT_THRESHOLD:
            severity = "severe"
        elif value >= ALERT_THRESHOLD:
            severity = "warning"
        else:
            continue
        alerts.append({
            "horizon": horizon,
            "severity": severity,
            "value": info["value"],
            "category": info["category"],
            "message": (f"{'Severe alert' if severity == 'severe' else 'Warning'}: "
                        f"AQI forecast for {horizon} is {info['value']} ({info['category']}). "
                        f"{info['message']}"),
        })
    return alerts
