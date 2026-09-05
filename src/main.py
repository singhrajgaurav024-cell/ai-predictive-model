"""
main.py
-------
Entry point that wires the whole pipeline together:
    1. Load data                (DataLoader)
    2. Preprocess / engineer    (Preprocessor)
    3. Train/test split + train (PredictiveModel)
    4. Evaluate                 (Evaluator)
    5. Persist customers, run metadata, and per-customer predictions (MySQL)

Run from the project root:
    python -m src.main --algorithm random_forest --use-db
"""

import argparse
import json
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.model import PredictiveModel
from src.evaluator import Evaluator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "customer_churn.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the churn prediction model.")
    parser.add_argument(
        "--algorithm", default="random_forest",
        choices=["random_forest", "logistic_regression"],
        help="Which classification algorithm to use.",
    )
    parser.add_argument(
        "--use-db", action="store_true",
        help="Persist customers, run metadata, and predictions to MySQL.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    MODEL_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    # 1. Load ----------------------------------------------------------------
    print(f"[1/5] Loading dataset from {DATA_PATH} ...")
    df = DataLoader(DATA_PATH).load()
    print(f"      {len(df)} rows loaded. Churn rate: {df['churn'].mean():.2%}")

    # 2. Preprocess ------------------------------------------------------------
    print("[2/5] Preprocessing features ...")
    pre = Preprocessor()
    X, y = pre.split_features_target(df)

    X_train_raw, X_test_raw, y_train, y_test, id_train, id_test = train_test_split(
        X, y, df["customer_id"], test_size=args.test_size, random_state=42, stratify=y
    )
    X_train = pre.fit_transform(X_train_raw)
    X_test = pre.transform(X_test_raw)

    # 3. Train -----------------------------------------------------------------
    print(f"[3/5] Training '{args.algorithm}' model on {X_train.shape[0]} samples ...")
    model = PredictiveModel(algorithm=args.algorithm)
    model.train(X_train, y_train)
    model.save(MODEL_DIR / f"{args.algorithm}.joblib")

    # 4. Evaluate --------------------------------------------------------------
    print(f"[4/5] Evaluating on {X_test.shape[0]} held-out samples ...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    evaluator = Evaluator(y_test, y_pred, y_proba)
    metrics = evaluator.compute_metrics()
    evaluator.plot_confusion_matrix(REPORTS_DIR / "confusion_matrix.png")
    evaluator.plot_roc_curve(REPORTS_DIR / "roc_curve.png")

    print("      Metrics:")
    for k, v in metrics.items():
        print(f"        {k:>10}: {v}")

    top_features = model.feature_importances(pre.get_feature_names())
    if top_features:
        print("      Top 5 features:")
        for name, importance in top_features[:5]:
            print(f"        {name:<35} {importance:.4f}")

    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(
            {
                "algorithm": args.algorithm,
                "train_size": int(X_train.shape[0]),
                "test_size": int(X_test.shape[0]),
                **metrics,
                "top_features": [{"feature": n, "importance": round(float(i), 4)}
                                 for n, i in (top_features or [])[:10]],
            },
            f, indent=2,
        )

    # 5. Persist to MySQL --------------------------------------------------------
    if args.use_db:
        print("[5/5] Persisting customers, run metadata, and predictions to MySQL ...")
        from database.db_manager import DatabaseManager

        with DatabaseManager() as db:
            db.initialize_schema(PROJECT_ROOT / "database" / "schema.sql")
            db.upsert_customers(df)
            run_id = db.log_model_run(
                algorithm=args.algorithm,
                train_size=X_train.shape[0],
                test_size=X_test.shape[0],
                metrics=metrics,
                notes="Automated run via src/main.py",
            )
            db.save_predictions(run_id, id_test.tolist(), y_pred, y_proba)
            print(f"      Run #{run_id} logged. Predictions stored for {len(id_test)} customers.")
    else:
        print("[5/5] Skipped MySQL persistence (pass --use-db to enable).")

    print("\nDone. Reports written to:", REPORTS_DIR)


if __name__ == "__main__":
    main()
