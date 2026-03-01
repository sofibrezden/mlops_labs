import os
import click
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def preprocess_data(df):

    df["datetime"] = pd.to_datetime(df["datetime"])

    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.weekday

    df = df.drop(["datetime", "casual", "registered"], axis=1)

    X = df.drop("count", axis=1)
    y = df["count"]

    return X, y


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.figure()
    plt.title("Feature Importance")
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.tight_layout()

    plt.savefig("feature_importance.png")
    plt.close()


@click.command()
@click.option("--data-path", default="data/raw/combined.csv", help="Path to dataset")
@click.option("--n-estimators", default=200, type=int)
@click.option("--min-samples-leaf", default=5, type=int)
@click.option("--max-depth", default=None, type=int)
@click.option("--random-state", default=42, type=int)
def main(
    data_path,
    n_estimators,
    max_depth,
    min_samples_leaf,
    random_state,
):

    mlflow.set_experiment("Bike_Sharing_RF_Experiment")

    df = pd.read_csv(data_path)
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    with mlflow.start_run(
        run_name=f"RF_depth={max_depth}_trees={n_estimators}_min_samples_leaf={min_samples_leaf}_random_state={random_state}"
    ):  

        mlflow.set_tag("model_type", "RandomForestRegressor")
        mlflow.set_tag("dataset", "Bike Sharing Demand")

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("random_state", random_state)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=random_state
        )

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

        mae_train = mean_absolute_error(y_train, y_train_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)

        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        mlflow.log_metric("rmse_train", rmse_train)
        mlflow.log_metric("rmse_test", rmse_test)

        mlflow.log_metric("mae_train", mae_train)
        mlflow.log_metric("mae_test", mae_test)

        mlflow.log_metric("r2_train", r2_train)
        mlflow.log_metric("r2_test", r2_test)

        os.makedirs("models", exist_ok=True)
        model_name = f"rf_depth={max_depth}_trees={n_estimators}"
        model_path = f"models/{model_name}"
        joblib.dump(model, model_path + ".pkl")

        mlflow.sklearn.log_model(model, model_name)

        plot_feature_importance(model, X.columns)
        mlflow.log_artifact("feature_importance.png")


if __name__ == "__main__":
    main()