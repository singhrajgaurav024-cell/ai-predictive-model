"""
test_pipeline.py
-----------------
Basic unit tests covering the core pipeline components. Run with:
    pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.model import PredictiveModel
from src.evaluator import Evaluator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "customer_churn.csv"


@pytest.fixture(scope="module")
def raw_df():
    return DataLoader(DATA_PATH).load()


def test_data_loader_loads_expected_columns(raw_df):
    expected = {
        "customer_id", "tenure_months", "monthly_charges", "total_charges",
        "contract_type", "internet_service", "tech_support", "online_security",
        "payment_method", "num_support_calls", "senior_citizen", "partner",
        "paperless_billing", "churn",
    }
    assert expected.issubset(set(raw_df.columns))


def test_data_loader_raises_on_missing_file(tmp_path):
    loader = DataLoader(tmp_path / "does_not_exist.csv")
    with pytest.raises(FileNotFoundError):
        loader.load()


def test_data_loader_raises_on_missing_target(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"customer_id": ["A1"], "tenure_months": [5]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError):
        DataLoader(bad_csv).load()


def test_preprocessor_split_and_transform_shapes(raw_df):
    pre = Preprocessor()
    X, y = pre.split_features_target(raw_df)
    assert len(X) == len(y)
    assert "churn" not in X.columns

    X_transformed = pre.fit_transform(X)
    assert X_transformed.shape[0] == len(X)
    assert X_transformed.shape[1] == len(pre.get_feature_names())


def test_model_train_predict_roundtrip(raw_df):
    pre = Preprocessor()
    X, y = pre.split_features_target(raw_df)
    X_transformed = pre.fit_transform(X)

    model = PredictiveModel(algorithm="logistic_regression")
    model.train(X_transformed, y)
    preds = model.predict(X_transformed)
    proba = model.predict_proba(X_transformed)

    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1})
    assert ((proba >= 0) & (proba <= 1)).all()


def test_model_predict_before_train_raises():
    model = PredictiveModel(algorithm="random_forest")
    with pytest.raises(RuntimeError):
        model.predict([[0, 0, 0]])


def test_model_rejects_unsupported_algorithm():
    with pytest.raises(ValueError):
        PredictiveModel(algorithm="not_a_real_algorithm")


def test_evaluator_metrics_are_bounded():
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.6, 0.7, 0.3])

    metrics = Evaluator(y_true, y_pred, y_proba).compute_metrics()
    for value in metrics.values():
        assert 0.0 <= value <= 1.0
