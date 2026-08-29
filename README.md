# 🌫️ Karachi AQI Forecasting System

An end-to-end, serverless machine learning pipeline that forecasts Karachi's Air Quality Index (AQI) 24, 48, and 72 hours ahead — from live data ingestion through to a public, interactive dashboard.

**Live demo:** [aqi-karachi-forecasting.streamlit.app](https://aqi-karachi-forecasting.streamlit.app/)

Built as part of the **10Pearls Shine Internship Program** — Data Sciences Domain, Cohort 9 (13 July – 4 September 2026) by **Alishba Jawaid**, 7th Semester BS (Artificial Intelligence).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [Automation (CI/CD)](#automation-cicd)
- [Deployment](#deployment)
- [Limitations & Roadmap](#limitations--roadmap)
- [Acknowledgements](#acknowledgements)

---

## Overview

Karachi regularly experiences AQI levels in the "Unhealthy" range and worse. This project treats AQI forecasting as a time-series regression problem and builds a **production-style system** around it — not a one-off notebook — covering:

- Automated hourly + one-time-backfill data collection from Open-Meteo
- A rich, domain-informed feature set (lags, rolling windows, cyclical encoding, upwind signals, weather trends)
- A Hopsworks **Feature Store** and **Model Registry** to decouple data engineering, training, and serving
- Multi-model training and evaluation (Ridge / Random Forest / XGBoost) across 3 forecast horizons
- A FastAPI backend + Streamlit dashboard with live predictions, EDA, SHAP explainability, and health alerts
- Free, fully automated deployment on Streamlit Community Cloud

## Architecture

<p align="center">
  <img src="docs/architecture.png" alt="System architecture diagram" width="850">
</p>

Two scheduled GitHub Actions jobs keep the system self-sustaining:

| Job | Script(s) | Schedule | Purpose |
|---|---|---|---|
| Feature refresh | `hourly_feature_update.py` | Hourly | Upserts the latest weather/AQI data into the Feature Store |
| Retraining | `refresh_training_split.py` → `train_models.py` | Daily | Refreshes the train/test split and retrains + re-registers the 3 winning models |

The **Feature Store** is the single source of truth read by both training and real-time inference (no train/serve skew). The **Model Registry** means the deployed app never bundles model files — `model_service.py` always downloads the current best-registered model per horizon at runtime.

## Features

- **Forecast tab** — current AQI + 24h/48h/72h predictions with EPA category badges and a trend chart
- **EDA & Trends tab** — summary stats, daily trend, hour-of-day pattern, monthly seasonality
- **Model Explainability tab** — local + global SHAP feature attribution per horizon
- **Alerts tab** — automatic warning/severe alerts when a forecast crosses hazardous AQI thresholds

## Tech Stack

| Layer | Technology |
|---|---|
| Data source | Open-Meteo (Weather, Air Quality, Geocoding APIs) |
| Feature engineering | pandas, numpy |
| Feature Store & Model Registry | Hopsworks |
| Modeling | scikit-learn, XGBoost (TensorFlow optional) |
| Explainability | SHAP |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit, Plotly |
| Automation | GitHub Actions |
| Hosting | Streamlit Community Cloud |

## Model Performance

Three candidates (Ridge, Random Forest, XGBoost) are trained independently per horizon; the lowest-RMSE model on a chronological, calendar-anchored 12-month test set wins. All three current winners are tuned Ridge Regression models, evaluated against a naive persistence baseline:

| Horizon | Model | Test RMSE | Test MAE | Test R² | Baseline R² | Improvement |
|---|---|---|---|---|---|---|
| 24h | Ridge | 12.46 | 8.72 | 0.67 | 0.432 | +0.237 |
| 48h | Ridge | 16.86 | 12.31 | 0.41 | 0.101 | +0.302 |
| 72h | Ridge | 17.82 | 13.30 | 0.342 | −0.093 | +0.427 |

The model's advantage over naive persistence **widens** as the horizon grows — exactly where a naive forecast is least useful.

## Project Structure

```
.
├── aqiPipeline.py              # Full historical backfill + feature engineering + Feature Store upload
├── hourly_feature_update.py    # Incremental hourly feature refresh (reuses aqiPipeline.py)
├── hopsworks_read_utils.py     # Resilient Feature Store reads (Hive-path fallback + retry/backoff)
├── refresh_training_split.py   # Daily: refreshes the Feature View train/test split
├── train_models.py             # Trains, evaluates, and registers models (Ridge/RF/XGBoost × 3 horizons)
├── dashboard_config.py         # Shared config constants (city, horizons, AQI breakpoints, API URL)
├── model_service.py            # Hopsworks connection caching + real-time prediction logic
├── eda_utils.py                # Exploratory data analysis helpers
├── shap_utils.py                # SHAP local/global explanation computation
├── alerts.py                   # AQI classification + threshold-based alerting
├── api_backend.py              # FastAPI REST backend
├── streamlit_app.py            # Streamlit dashboard (+ backend subprocess launcher for deployment)
├── requirements.txt            # Slim deps for the deployed dashboard (read by Streamlit Cloud)
├── requirements-pipeline.txt   # Full deps for the GitHub Actions pipelines
└── docs/
    └── architecture.png        # Architecture diagram used in this README
```

## Getting Started

### Prerequisites

- Python 3.10+
- A free [Hopsworks](https://www.hopsworks.ai/) account (Feature Store + Model Registry)

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements-pipeline.txt           # full deps, needed to run the pipelines locally
```

### 2. Configure credentials

Create a `.env` file in the repo root (never commit this file):

```env
HOPSWORKS_API_KEY=your_api_key
HOPSWORKS_HOST=your_hopsworks_host
HOPSWORKS_PROJECT=your_project_name
```

### 3. Build the dataset & train models (first-time setup)

```bash
python aqiPipeline.py --city "Karachi"     # one-time historical backfill + Feature Store upload
python train_models.py --city karachi      # trains & registers the first set of models
```

### 4. Run the app locally

```bash
uvicorn api_backend:app --reload --port 8000   # terminal 1
streamlit run streamlit_app.py                 # terminal 2
```

Visit `http://localhost:8501`.

## Environment Variables

| Variable | Used by | Description |
|---|---|---|
| `HOPSWORKS_API_KEY` | all pipeline + serving scripts | Hopsworks project API key |
| `HOPSWORKS_HOST` | all pipeline + serving scripts | Hopsworks cluster hostname |
| `HOPSWORKS_PROJECT` | all pipeline + serving scripts | Hopsworks project name |
| `API_BASE_URL` | `streamlit_app.py` (optional) | Override if the backend runs on a different host/port than `127.0.0.1:8000` |

In production, these are set via GitHub Actions Secrets (pipelines) and the Streamlit Cloud Secrets manager (dashboard) — never committed to the repo.

## Automation (CI/CD)

Two scheduled GitHub Actions workflows keep the system current without manual intervention:

- **Hourly** — `hourly_feature_update.py` keeps the Feature Store up to date with the latest weather/AQI readings.
- **Daily** — `refresh_training_split.py` followed by `train_models.py` refreshes the train/test split and retrains + re-registers the 3 winning models, so forecasts don't drift stale as seasons change.

Both install dependencies from `requirements-pipeline.txt`.

## Deployment

The dashboard is deployed free on **Streamlit Community Cloud**, auto-redeploying on every push. Because the app is actually two services (FastAPI backend + Streamlit frontend) running on a platform that gives one process per app, `streamlit_app.py` spawns `uvicorn api_backend:app` as a background subprocess on first load. Streamlit Cloud reads dependencies from the slim `requirements.txt` at the repo root — the full pipeline dependency list lives separately in `requirements-pipeline.txt` so GitHub Actions and the live deployment never conflict.

## Limitations & Roadmap

- Currently single-city (Karachi); extending to other cities is a config change, not yet tested end-to-end
- No automated test suite around the pipelines/API yet
- No monitoring/alerting on pipeline run failures themselves
- Optional Keras neural-network candidate exists but isn't the primary model comparison
- Docker-based deployment remains a good option once available, removing the need for the subprocess-launcher workaround

## Acknowledgements

- [Open-Meteo](https://open-meteo.com/) for free weather and air-quality data
- [Hopsworks](https://www.hopsworks.ai/) for the Feature Store & Model Registry
- 10Pearls Shine Internship Program, Data Sciences Domain — Cohort 9

---

**Author:** Alishba Jawaid — 7th Semester, BS (Artificial Intelligence)
**Live app:** https://aqi-karachi-forecasting.streamlit.app/
