"""
generate_dataset.py
--------------------
Generates a realistic, synthetic customer-churn dataset used to train and
evaluate the predictive model. Ships with the project so the pipeline is
fully reproducible without needing an external data source.

Run:
    python data/generate_dataset.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_SAMPLES = 2000

np.random.seed(RANDOM_SEED)


def generate_customer_data(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    """Builds a synthetic telecom-style customer dataset with a realistic
    relationship between features and the churn label."""

    customer_id = [f"CUST{i:05d}" for i in range(1, n_samples + 1)]
    tenure_months = np.random.gamma(shape=2.0, scale=12, size=n_samples).clip(0, 72).round(0)
    monthly_charges = np.random.normal(70, 25, n_samples).clip(15, 150).round(2)
    total_charges = (monthly_charges * tenure_months * np.random.uniform(0.85, 1.05, n_samples)).round(2)
    contract_type = np.random.choice(
        ["Month-to-Month", "One Year", "Two Year"], size=n_samples, p=[0.55, 0.25, 0.20]
    )
    internet_service = np.random.choice(
        ["DSL", "Fiber Optic", "No Internet"], size=n_samples, p=[0.35, 0.45, 0.20]
    )
    tech_support = np.random.choice(["Yes", "No"], size=n_samples, p=[0.4, 0.6])
    online_security = np.random.choice(["Yes", "No"], size=n_samples, p=[0.35, 0.65])
    payment_method = np.random.choice(
        ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"],
        size=n_samples,
    )
    num_support_calls = np.random.poisson(1.5, n_samples).clip(0, 10)
    senior_citizen = np.random.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], size=n_samples, p=[0.5, 0.5])
    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.6, 0.4])

    # Build churn probability from a weighted combination of realistic risk
    # factors so the label is genuinely learnable (not random noise).
    risk = np.zeros(n_samples)
    risk += (contract_type == "Month-to-Month") * 0.55
    risk += (contract_type == "One Year") * 0.12
    risk += (internet_service == "Fiber Optic") * 0.22
    risk += (tech_support == "No") * 0.22
    risk += (online_security == "No") * 0.18
    risk += (payment_method == "Electronic Check") * 0.18
    risk += (num_support_calls >= 4) * 0.30
    risk += (tenure_months < 6) * 0.40
    risk += (tenure_months > 48) * -0.35
    risk += (monthly_charges > 90) * 0.15
    risk += -0.01 * tenure_months  # smooth tenure effect
    risk += np.random.normal(0, 0.05, n_samples)  # small noise for realism

    churn_prob = 1 / (1 + np.exp(-(risk * 4.5 - 3.3)))  # squash to (0,1)
    churn = (np.random.uniform(0, 1, n_samples) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "tenure_months": tenure_months.astype(int),
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "tech_support": tech_support,
            "online_security": online_security,
            "payment_method": payment_method,
            "num_support_calls": num_support_calls,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "paperless_billing": paperless_billing,
            "churn": churn,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_customer_data()
    out_path = Path(__file__).parent / "customer_churn.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(f"Churn rate: {df['churn'].mean():.2%}")
