-- schema.sql
-- Relational schema for the AI-Based Predictive Model project.
-- Stores customer records and every model-run's evaluation results so
-- performance can be tracked and compared across experiments.

CREATE DATABASE IF NOT EXISTS predictive_model_db;
USE predictive_model_db;

-- Raw customer records used for training / prediction
CREATE TABLE IF NOT EXISTS customers (
    customer_id         VARCHAR(20)  PRIMARY KEY,
    tenure_months       INT          NOT NULL,
    monthly_charges     DECIMAL(8,2) NOT NULL,
    total_charges       DECIMAL(10,2) NOT NULL,
    contract_type       VARCHAR(20)  NOT NULL,
    internet_service    VARCHAR(20)  NOT NULL,
    tech_support        VARCHAR(5)   NOT NULL,
    online_security     VARCHAR(5)   NOT NULL,
    payment_method      VARCHAR(30)  NOT NULL,
    num_support_calls   INT          NOT NULL,
    senior_citizen      TINYINT(1)   NOT NULL,
    partner             VARCHAR(5)   NOT NULL,
    paperless_billing   VARCHAR(5)   NOT NULL,
    churn_actual         TINYINT(1)  NULL,
    INDEX idx_contract (contract_type)
) ENGINE=InnoDB;

-- One row per training/evaluation run, for experiment tracking
CREATE TABLE IF NOT EXISTS model_runs (
    run_id          INT AUTO_INCREMENT PRIMARY KEY,
    run_timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
    algorithm       VARCHAR(50)  NOT NULL,
    train_size      INT          NOT NULL,
    test_size       INT          NOT NULL,
    accuracy        DECIMAL(6,4) NOT NULL,
    precision_score DECIMAL(6,4) NOT NULL,
    recall_score    DECIMAL(6,4) NOT NULL,
    f1_score        DECIMAL(6,4) NOT NULL,
    roc_auc         DECIMAL(6,4) NOT NULL,
    notes           VARCHAR(255)
) ENGINE=InnoDB;

-- Per-customer predictions for a given run, for auditability
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id    INT AUTO_INCREMENT PRIMARY KEY,
    run_id           INT NOT NULL,
    customer_id      VARCHAR(20) NOT NULL,
    predicted_churn  TINYINT(1) NOT NULL,
    churn_probability DECIMAL(6,4) NOT NULL,
    FOREIGN KEY (run_id) REFERENCES model_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE,
    INDEX idx_run (run_id)
) ENGINE=InnoDB;
