# Bike Sharing Demand Prediction

Machine learning project for predicting bike rental demand using regression and time series analysis.

## Dataset

[Bike Sharing Demand - Kaggle Competition](https://www.kaggle.com/c/bike-sharing-demand)

## Problem Statement

Predict hourly bike rental demand based on historical usage patterns and environmental factors.

## Key Features

- **Task Type**: Regression, Time Series Forecasting
- **Time Series Analysis**: Temporal patterns and trends
- **Seasonality**: Daily, weekly, and seasonal variations
- **Weather Impact**: Temperature, humidity, wind speed effects

## Project Structure

```
mlops_lab_1/
├── data/
│   └── raw/
│       └── dataset.csv
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   └── train.py
└── requirements.txt
└── .gitignore
```

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Getting Started

1. Download the dataset from Kaggle
2. Place data files in `data/raw/`
3. Run exploratory data analysis in `notebooks/01_eda.ipynb`
