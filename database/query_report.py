"""
query_report.py
---------------
Demonstrates querying the MySQL database for insights: run history and
the highest-risk customers from the most recent model run.

Run (after src/main.py has been run at least once with --use-db):
    python database/query_report.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from database.db_manager import DatabaseManager


def main():
    with DatabaseManager() as db:
        history = db.get_run_history()
        if history.empty:
            print("No model runs found. Run `python -m src.main --use-db` first.")
            return

        print("=== Model Run History ===")
        print(history[["run_id", "algorithm", "accuracy", "precision_score",
                        "recall_score", "f1_score", "roc_auc", "run_timestamp"]]
              .to_string(index=False))

        latest_run_id = int(history.iloc[0]["run_id"])
        print(f"\n=== Top 10 Highest-Risk Customers (Run #{latest_run_id}) ===")
        high_risk = db.get_high_risk_customers(latest_run_id, threshold=0.6)
        if high_risk.empty:
            print("No customers above the risk threshold in this run.")
        else:
            print(high_risk.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
