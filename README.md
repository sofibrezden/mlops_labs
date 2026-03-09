# Bike Sharing Demand Prediction

Machine learning project for predicting bike rental demand using regression. The pipeline is orchestrated with DVC and experiments are tracked with MLflow.

## Dataset

[Bike Sharing Demand - Kaggle Competition](https://www.kaggle.com/c/bike-sharing-demand)

## Problem

Predict hourly bike rental demand from historical usage and environmental features (weather, time of day, seasonality).

## Project Structure

```
mlops_lab_2/
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── prepare.py        # data preprocessing and train/test split
│   └── train.py          # Random Forest training and MLflow logging
├── dvc.yaml              # DVC pipeline definition
├── dvc.lock              # locked pipeline state
├── requirements.txt
├── .gitignore
└── .dvcignore
```

## Requirements

```bash
pip install -r requirements.txt
```

## Pipeline (DVC)

Two stages:

1. **prepare** – reads raw CSV, preprocesses (datetime features, drop columns), splits into train/test (80/20, temporal order) and writes `data/prepared/train.csv` and `test.csv`.
2. **train** – trains a Random Forest regressor on prepared data, logs runs to MLflow, saves model and feature importance plot under `data/models`.

Reproduce the full pipeline:

```bash
dvc repro
```

Run a single stage:

```bash
dvc repro prepare
dvc repro train
```

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Place the Kaggle dataset as `data/raw/dataset.csv` (or track it with DVC).
3. Run the pipeline: `dvc repro`
4. Optionally run EDA in `notebooks/01_eda.ipynb`.

## MLflow

Training runs are logged to MLflow (parameters, metrics, artifacts). Use the MLflow UI to compare runs:

```bash
mlflow ui
```

Then open the URL shown (default `http://127.0.0.1:5000`).
