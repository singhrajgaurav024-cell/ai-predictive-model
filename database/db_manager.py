"""
db_manager.py
-------------
Encapsulates all MySQL access behind a small, purpose-built API so the
rest of the codebase never writes raw SQL. Structuring the schema this
way (customers / model_runs / predictions) keeps lookups fast and keeps
every experiment's results queryable and comparable later.

Configure the connection with environment variables (falls back to the
defaults used for local development):
    DB_HOST, DB_USER, DB_PASSWORD, DB_NAME
"""

import os
from pathlib import Path

import mysql.connector
import pandas as pd
from mysql.connector import Error as MySQLError


class DatabaseManager:
    def __init__(
        self,
        host: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        self.host = host or os.getenv("DB_HOST", "localhost")
        self.user = user or os.getenv("DB_USER", "ml_app")
        self.password = password or os.getenv("DB_PASSWORD", "MlApp@2025")
        self.database = database or os.getenv("DB_NAME", "predictive_model_db")
        self.connection = None

    # -- connection lifecycle -------------------------------------------------
    def connect(self) -> None:
        self.connection = mysql.connector.connect(
            host=self.host, user=self.user, password=self.password, database=self.database
        )

    def close(self) -> None:
        if self.connection and self.connection.is_connected():
            self.connection.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # -- schema setup ----------------------------------------------------------
    def initialize_schema(self, schema_path: str | Path) -> None:
        """Executes schema.sql to (re)create tables. Safe to re-run (uses
        CREATE TABLE IF NOT EXISTS)."""
        with open(schema_path, "r") as f:
            statements = f.read().split(";")
        cursor = self.connection.cursor()
        for statement in statements:
            statement = statement.strip()
            if statement:
                cursor.execute(statement)
        self.connection.commit()
        cursor.close()

    # -- customers ---------------------------------------------------------
    def upsert_customers(self, df: pd.DataFrame) -> int:
        cursor = self.connection.cursor()
        rows = df.to_dict("records")
        query = """
            INSERT INTO customers (
                customer_id, tenure_months, monthly_charges, total_charges,
                contract_type, internet_service, tech_support, online_security,
                payment_method, num_support_calls, senior_citizen, partner,
                paperless_billing, churn_actual
            ) VALUES (
                %(customer_id)s, %(tenure_months)s, %(monthly_charges)s, %(total_charges)s,
                %(contract_type)s, %(internet_service)s, %(tech_support)s, %(online_security)s,
                %(payment_method)s, %(num_support_calls)s, %(senior_citizen)s, %(partner)s,
                %(paperless_billing)s, %(churn)s
            )
            ON DUPLICATE KEY UPDATE
                tenure_months = VALUES(tenure_months),
                monthly_charges = VALUES(monthly_charges),
                total_charges = VALUES(total_charges),
                churn_actual = VALUES(churn_actual)
        """
        cursor.executemany(query, rows)
        self.connection.commit()
        affected = cursor.rowcount
        cursor.close()
        return affected

    # -- model runs ----------------------------------------------------------
    def log_model_run(self, algorithm: str, train_size: int, test_size: int,
                       metrics: dict, notes: str = "") -> int:
        cursor = self.connection.cursor()
        query = """
            INSERT INTO model_runs (
                algorithm, train_size, test_size, accuracy,
                precision_score, recall_score, f1_score, roc_auc, notes
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (
            algorithm, train_size, test_size,
            metrics["accuracy"], metrics["precision"], metrics["recall"],
            metrics["f1_score"], metrics["roc_auc"], notes,
        ))
        self.connection.commit()
        run_id = cursor.lastrowid
        cursor.close()
        return run_id

    # -- predictions ---------------------------------------------------------
    def save_predictions(self, run_id: int, customer_ids, predictions, probabilities) -> int:
        cursor = self.connection.cursor()
        query = """
            INSERT INTO predictions (run_id, customer_id, predicted_churn, churn_probability)
            VALUES (%s, %s, %s, %s)
        """
        rows = [
            (run_id, cid, int(pred), float(prob))
            for cid, pred, prob in zip(customer_ids, predictions, probabilities)
        ]
        cursor.executemany(query, rows)
        self.connection.commit()
        affected = cursor.rowcount
        cursor.close()
        return affected

    # -- reporting queries ---------------------------------------------------
    def get_run_history(self) -> pd.DataFrame:
        query = "SELECT * FROM model_runs ORDER BY run_timestamp DESC"
        return pd.read_sql(query, self.connection)

    def get_high_risk_customers(self, run_id: int, threshold: float = 0.7) -> pd.DataFrame:
        query = """
            SELECT c.customer_id, c.contract_type, c.tenure_months,
                   p.churn_probability
            FROM predictions p
            JOIN customers c ON c.customer_id = p.customer_id
            WHERE p.run_id = %s AND p.churn_probability >= %s
            ORDER BY p.churn_probability DESC
        """
        return pd.read_sql(query, self.connection, params=(run_id, threshold))
