"""
data_loader.py
--------------
Handles all dataset I/O. Encapsulating loading logic in its own class keeps
main.py free of file-path/format details (single-responsibility principle).
"""

from pathlib import Path
import pandas as pd


class DataLoader:
    """Loads the raw dataset from disk and performs basic sanity checks."""

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)

    def load(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.csv_path}. "
                f"Run `python data/generate_dataset.py` first."
            )
        df = pd.read_csv(self.csv_path)
        self._validate(df)
        return df

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Loaded dataset is empty.")
        if df.isnull().values.any():
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            raise ValueError(f"Dataset contains missing values:\n{missing}")
        if "churn" not in df.columns:
            raise ValueError("Expected target column 'churn' not found in dataset.")
