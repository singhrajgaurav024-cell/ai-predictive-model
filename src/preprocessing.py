"""
preprocessing.py
----------------
Feature engineering and preprocessing, encapsulated so the exact same
transformation can be re-applied at training and at inference time
(prevents train/serve skew).
"""

from dataclasses import dataclass, field

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class Preprocessor:
    """Builds and stores the fitted preprocessing pipeline (scaling +
    one-hot encoding) so it can be reused consistently."""

    numeric_features: list[str] = field(
        default_factory=lambda: [
            "tenure_months",
            "monthly_charges",
            "total_charges",
            "num_support_calls",
            "senior_citizen",
        ]
    )
    categorical_features: list[str] = field(
        default_factory=lambda: [
            "contract_type",
            "internet_service",
            "tech_support",
            "online_security",
            "payment_method",
            "partner",
            "paperless_billing",
        ]
    )
    target_column: str = "churn"
    id_column: str = "customer_id"

    def __post_init__(self):
        self.column_transformer: ColumnTransformer | None = None

    def split_features_target(self, df: pd.DataFrame):
        y = df[self.target_column]
        X = df.drop(columns=[self.target_column, self.id_column])
        return X, y

    def build_transformer(self) -> ColumnTransformer:
        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", drop="first"),
                    self.categorical_features,
                ),
            ]
        )
        return self.column_transformer

    def fit_transform(self, X: pd.DataFrame):
        if self.column_transformer is None:
            self.build_transformer()
        return self.column_transformer.fit_transform(X)

    def transform(self, X: pd.DataFrame):
        if self.column_transformer is None:
            raise RuntimeError("Transformer not fitted yet. Call fit_transform first.")
        return self.column_transformer.transform(X)

    def get_feature_names(self) -> list[str]:
        return list(self.column_transformer.get_feature_names_out())
