"""
model.py
--------
Object-oriented wrapper around the classification algorithm. Wrapping
scikit-learn estimators in our own class keeps the training/prediction
interface stable even if the underlying algorithm is swapped out later.
"""

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class PredictiveModel:
    """A swappable classification model with a uniform train/predict/save
    interface, used to predict customer churn (1 = will churn, 0 = will not)."""

    SUPPORTED_ALGORITHMS = {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
        ),
        "logistic_regression": lambda: LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
    }

    def __init__(self, algorithm: str = "random_forest"):
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. "
                f"Choose from {list(self.SUPPORTED_ALGORITHMS)}."
            )
        self.algorithm_name = algorithm
        self.estimator = self.SUPPORTED_ALGORITHMS[algorithm]()
        self._is_fitted = False

    def train(self, X_train, y_train) -> "PredictiveModel":
        self.estimator.fit(X_train, y_train)
        self._is_fitted = True
        return self

    def predict(self, X):
        self._check_fitted()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        self._check_fitted()
        return self.estimator.predict_proba(X)[:, 1]

    def feature_importances(self, feature_names: list[str]):
        self._check_fitted()
        if not hasattr(self.estimator, "feature_importances_"):
            return None
        pairs = sorted(
            zip(feature_names, self.estimator.feature_importances_),
            key=lambda p: p[1],
            reverse=True,
        )
        return pairs

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        joblib.dump(self.estimator, path)

    @classmethod
    def load(cls, path: str | Path, algorithm: str = "random_forest") -> "PredictiveModel":
        instance = cls(algorithm=algorithm)
        instance.estimator = joblib.load(path)
        instance._is_fitted = True
        return instance

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Model has not been trained yet. Call .train() first.")
