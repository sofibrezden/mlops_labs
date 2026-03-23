import pytest
import os


class TestImports:

    def test_import_pandas(self):
        import pandas as pd

        assert pd is not None

    def test_import_numpy(self):
        import numpy as np

        assert np is not None

    def test_import_sklearn(self):
        from sklearn.ensemble import RandomForestRegressor

        assert RandomForestRegressor is not None

    def test_import_mlflow(self):
        import mlflow

        assert mlflow is not None

    def test_import_joblib(self):
        import joblib

        assert joblib is not None

    def test_import_optuna(self):
        import optuna

        assert optuna is not None

    def test_import_hydra(self):
        import hydra

        assert hydra is not None


class TestProjectStructure:

    def test_src_directory_exists(self):
        assert os.path.exists("src"), "src directory not found"

    def test_data_directory_exists(self):
        assert os.path.exists("data"), "data directory not found"

    def test_config_directory_exists(self):
        assert os.path.exists("config"), "config directory not found"

    def test_train_script_exists(self):
        assert os.path.exists("src/train.py"), "train.py script not found"

    def test_prepare_script_exists(self):
        assert os.path.exists("src/prepare.py"), "prepare.py script not found"

    def test_optimize_script_exists(self):
        assert os.path.exists("src/optimize.py"), "optimize.py script not found"

    def test_dvc_yaml_exists(self):
        assert os.path.exists("dvc.yaml"), "dvc.yaml not found"

    def test_config_yaml_exists(self):
        assert os.path.exists("config/config.yaml"), "config.yaml not found"


class TestScriptsSyntax:

    def test_train_script_syntax(self):
        import py_compile

        try:
            py_compile.compile("src/train.py", doraise=True)
        except py_compile.PyCompileError as e:
            pytest.fail(f"Syntax error in train.py: {str(e)}")

    def test_prepare_script_syntax(self):
        import py_compile

        try:
            py_compile.compile("src/prepare.py", doraise=True)
        except py_compile.PyCompileError as e:
            pytest.fail(f"Syntax error in prepare.py: {str(e)}")

    def test_optimize_script_syntax(self):
        import py_compile

        try:
            py_compile.compile("src/optimize.py", doraise=True)
        except py_compile.PyCompileError as e:
            pytest.fail(f"Syntax error in optimize.py: {str(e)}")
