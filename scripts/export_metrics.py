import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def export_metrics():
    model_path = None
    if os.path.exists("models/best_model.pkl"):
        model_path = "models/best_model.pkl"
    elif os.path.exists("data/models/model.pkl"):
        model_path = "data/models/model.pkl"
    else:
        print("No model found to export metrics")
        return

    test_path = "data/prepared/test.csv"
    if not os.path.exists(test_path):
        print("Test data not found")
        return

    model = joblib.load(model_path)

    test_df = pd.read_csv(test_path)
    X_test = test_df.drop("count", axis=1)
    y_test = test_df["count"]

    y_pred = model.predict(X_test)

    metrics = {
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae_test": float(mean_absolute_error(y_test, y_pred)),
        "r2_test": float(r2_score(y_test, y_pred)),
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Metrics exported to metrics.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    export_metrics()
