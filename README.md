# AI-Based Predictive Model — Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely
to churn (cancel their subscription), built with Python, scikit-learn, and a
MySQL backend for data persistence and experiment tracking.

> Built to match the "AI-Based Predictive Model" academic mini-project entry
> on Raj Gaurav Singh's resume: a Python ML classifier, a relational MySQL
> schema for storing project data, an object-oriented/modular codebase
> version-controlled with Git, and a documented evaluation workflow.

## What it does

1. Generates (or accepts) a customer dataset with 13 features — tenure,
   billing, contract type, service usage, support-call history, etc.
2. Cleans and encodes the data (scaling + one-hot encoding).
3. Trains a classifier (Random Forest or Logistic Regression) to predict
   the `churn` label.
4. Evaluates the model (accuracy, precision, recall, F1, ROC-AUC) and
   generates a confusion matrix and ROC curve.
5. Persists customer records, model-run metadata, and per-customer
   predictions to a MySQL database so results are queryable and every
   experiment run is auditable and comparable over time.

## Results (current model)

| Metric | Score |
|---|---|
| Accuracy | 79.5% |
| Precision | 70.9% |
| Recall | 75.7% |
| F1 Score | 73.2% |
| ROC-AUC | 0.869 |

Top predictive features: **tenure**, **total charges**, **monthly charges**,
and **contract type** (month-to-month customers churn far more than
one/two-year contract holders) — see `reports/metrics.json` for the full
ranked list and `reports/roc_curve.png` / `reports/confusion_matrix.png` for
the plots.

## Project structure

```
ai_predictive_model/
├── data/
│   ├── generate_dataset.py     # Generates the synthetic customer dataset
│   └── customer_churn.csv      # Generated dataset (2,000 customers)
├── database/
│   ├── schema.sql              # MySQL schema: customers, model_runs, predictions
│   ├── db_manager.py           # DatabaseManager class — all SQL lives here
│   └── query_report.py         # Example queries: run history, high-risk customers
├── src/
│   ├── data_loader.py          # DataLoader class — loads & validates the CSV
│   ├── preprocessing.py        # Preprocessor class — scaling + one-hot encoding
│   ├── model.py                # PredictiveModel class — train/predict/save/load
│   ├── evaluator.py            # Evaluator class — metrics + plots
│   └── main.py                 # Orchestrates the full pipeline end-to-end
├── tests/
│   └── test_pipeline.py        # Unit tests (pytest) for every module
├── reports/                    # Generated: metrics.json, confusion_matrix.png, roc_curve.png
├── models/                     # Generated: saved .joblib model file
├── requirements.txt
└── README.md
```

The codebase is intentionally split into single-responsibility classes
(`DataLoader`, `Preprocessor`, `PredictiveModel`, `Evaluator`,
`DatabaseManager`) rather than one long script, so each piece can be tested,
reused, and swapped independently — e.g. `PredictiveModel` supports either
`random_forest` or `logistic_regression` without changing any other file.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up MySQL

Make sure a MySQL server is running, then create the database and a
dedicated application user (adjust the password as you like):

```sql
CREATE DATABASE IF NOT EXISTS predictive_model_db;
CREATE USER IF NOT EXISTS 'ml_app'@'localhost' IDENTIFIED BY 'MlApp@2025';
GRANT ALL PRIVILEGES ON predictive_model_db.* TO 'ml_app'@'localhost';
FLUSH PRIVILEGES;
```

The full table definitions (`customers`, `model_runs`, `predictions`) are in
`database/schema.sql` and are created automatically the first time you run
the pipeline with `--use-db` — you don't need to run the `.sql` file by hand.

By default the app connects as `ml_app` / `MlApp@2025` on `localhost`. To use
different credentials, set environment variables instead of editing code:

```bash
export DB_HOST=localhost
export DB_USER=ml_app
export DB_PASSWORD=your_password
export DB_NAME=predictive_model_db
```

### 3. Generate the dataset

```bash
python data/generate_dataset.py
```

### 4. Run the full pipeline

```bash
# Train + evaluate only (no database required)
python -m src.main --algorithm random_forest

# Train + evaluate + persist everything to MySQL
python -m src.main --algorithm random_forest --use-db

# Try the other algorithm
python -m src.main --algorithm logistic_regression --use-db
```

This prints the evaluation metrics to the console and writes:
- `models/<algorithm>.joblib` — the trained model
- `reports/metrics.json` — metrics + top feature importances
- `reports/confusion_matrix.png`
- `reports/roc_curve.png`

### 5. Query the results from MySQL

```bash
python database/query_report.py
```

Prints the history of every model run (so you can compare algorithms/experiments
over time) and the highest-risk customers from the latest run.

### 6. Run the tests

```bash
pytest tests/ -v
```

## Tech stack

- **Language:** Python 3
- **ML:** scikit-learn (RandomForestClassifier, LogisticRegression)
- **Data:** pandas, numpy
- **Visualization:** matplotlib
- **Database:** MySQL (via `mysql-connector-python`)
- **Testing:** pytest
- **Version control:** Git

## Notes on the dataset

`data/generate_dataset.py` produces a synthetic but realistic dataset: churn
probability is derived from a weighted combination of genuine risk factors
(short tenure, month-to-month contracts, high support-call volume, no tech
support/online security, electronic-check payment) plus random noise, so the
label is learnable but not trivial — the same shape of problem you'd get from
a real telecom churn dataset (e.g. IBM's public Telco Customer Churn dataset).
Swap in your own CSV at `data/customer_churn.csv` (same column names) to use
real data instead.
