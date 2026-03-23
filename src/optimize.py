import os
import random
import numpy as np
import joblib
import mlflow
import optuna
import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop("count", axis=1)
    y_train = train_df["count"]
    
    X_test = test_df.drop("count", axis=1)
    y_test = test_df["count"]
    
    return X_train, X_test, y_train, y_test

def build_model(params, seed):
    return RandomForestRegressor(
        random_state=seed,
        n_jobs=-1,
        **params
    )

def objective_factory(cfg, X_train, X_test, y_train, y_test):

    def objective(trial):

        if cfg.hpo.sampler == "grid":
            params = {
                "n_estimators": trial.suggest_categorical(
                    "n_estimators",
                    cfg.hpo.random_forest.n_estimators
                ),
                "max_depth": trial.suggest_categorical(
                    "max_depth",
                    cfg.hpo.random_forest.max_depth
                ),
                "min_samples_split": trial.suggest_categorical(
                    "min_samples_split",
                    cfg.hpo.random_forest.min_samples_split
                ),
                "min_samples_leaf": trial.suggest_categorical(
                    "min_samples_leaf",
                    cfg.hpo.random_forest.min_samples_leaf
                ),
            }
        else:
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    cfg.hpo.random_forest.n_estimators.low,
                    cfg.hpo.random_forest.n_estimators.high
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    cfg.hpo.random_forest.max_depth.low,
                    cfg.hpo.random_forest.max_depth.high
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split",
                    cfg.hpo.random_forest.min_samples_split.low,
                    cfg.hpo.random_forest.min_samples_split.high
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    cfg.hpo.random_forest.min_samples_leaf.low,
                    cfg.hpo.random_forest.min_samples_leaf.high
                ),
            }

        with mlflow.start_run(nested=True,
                              run_name=f"trial_{trial.number}"):

            mlflow.log_params(params)

            model = build_model(params, cfg.seed)
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

            score = rmse_test

            return score

    return objective

@hydra.main(version_base=None,
            config_path="../config",
            config_name="config")
def main(cfg: DictConfig):

    set_seed(cfg.seed)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    X_train, X_test, y_train, y_test = load_data(cfg.data.train_path, cfg.data.test_path)

    if cfg.hpo.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    elif cfg.hpo.sampler == "grid":
        search_space = {
            "n_estimators": cfg.hpo.random_forest.n_estimators,
            "max_depth": cfg.hpo.random_forest.max_depth,
            "min_samples_split": cfg.hpo.random_forest.min_samples_split,
            "min_samples_leaf": cfg.hpo.random_forest.min_samples_leaf,
        }
        sampler = optuna.samplers.GridSampler(search_space, seed=cfg.seed)
    else:
        sampler = optuna.samplers.RandomSampler(seed=cfg.seed)

    with mlflow.start_run(run_name=f"hpo_{cfg.hpo.sampler}") as parent:

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler
        )

        objective = objective_factory(cfg,
                                       X_train, X_test,
                                       y_train, y_test)

        study.optimize(objective,
                       n_trials=cfg.hpo.n_trials)

        best_params = study.best_params
        mlflow.log_dict(best_params, "best_params.json")
        mlflow.log_metric("best_rmse", study.best_value)

        best_model = build_model(best_params, cfg.seed)
        best_model.fit(X_train, y_train)

        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/best_model.pkl")
        mlflow.log_artifact("models/best_model.pkl")

        mlflow.sklearn.log_model(best_model,
                                 artifact_path="model")

if __name__ == "__main__":
    main()