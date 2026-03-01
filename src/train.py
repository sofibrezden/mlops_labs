import os
import click
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,  mean_absolute_error, r2_score


def plot_feature_importance(model, feature_names, output_path):
    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.figure()
    plt.title("Feature Importance")
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


@click.command()
@click.argument("input_dir")
@click.argument("model_dir")
@click.option("--n-estimators", type=int)
@click.option("--max-depth", type=int)
@click.option("--min-samples-leaf", type=int)
@click.option("--random-state", type=int)
def main(input_dir, model_dir, n_estimators, max_depth, min_samples_leaf, random_state):

    os.makedirs(model_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(input_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(input_dir, "test.csv"))

    X_train = train_df.drop("count", axis=1)
    y_train = train_df["count"]

    X_test = test_df.drop("count", axis=1)
    y_test = test_df["count"]

    mlflow.set_experiment("Bike_Sharing_RF_Experiment")

    with mlflow.start_run(run_name=f"RF_depth={max_depth}_trees={n_estimators}_min_samples_leaf={min_samples_leaf}_random_state={random_state}"):

        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state
        })

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
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

        model_path = os.path.join(model_dir, "model.pkl")
        joblib.dump(model, model_path)

        mlflow.sklearn.log_model(model, name="random_forest_model")
        mlflow.log_artifact(model_path)

        plot_path = os.path.join(model_dir, "feature_importance.png")
        plot_feature_importance(model, X_train.columns, plot_path)
        mlflow.log_artifact(plot_path)


if __name__ == "__main__":
    main()