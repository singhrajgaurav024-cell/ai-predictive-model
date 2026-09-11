from flask import Flask, request, render_template_string
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"

# Load trained model + preprocessing
pipeline = joblib.load(MODEL_PATH)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Customer Churn Prediction</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f2f4f7;
            margin: 0;
            padding: 40px;
        }

        .container {
            width: 700px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        }

        h1 {
            text-align: center;
            color: #222;
        }

        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 25px;
        }

        label {
            display: block;
            margin-top: 15px;
            font-weight: bold;
        }

        input, select {
            width: 100%;
            padding: 10px;
            margin-top: 6px;
            border: 1px solid #ccc;
            border-radius: 6px;
            box-sizing: border-box;
        }

        button {
            width: 100%;
            padding: 13px;
            margin-top: 25px;
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
        }

        button:hover {
            background: #1d4ed8;
        }

        .result {
            margin-top: 25px;
            padding: 20px;
            background: #eef6ff;
            border-radius: 8px;
            text-align: center;
        }

        .result h2 {
            margin-bottom: 10px;
        }

        .risk {
            font-size: 20px;
            font-weight: bold;
        }
    </style>
</head>

<body>

<div class="container">

    <h1>Customer Churn Prediction</h1>

    <p class="subtitle">
        AI-Based Predictive Model using Random Forest
    </p>

    <form method="POST">

        <label>Tenure (Months)</label>
        <input type="number" name="tenure_months" min="0" required>

        <label>Monthly Charges</label>
        <input type="number" step="0.01" name="monthly_charges" required>

        <label>Total Charges</label>
        <input type="number" step="0.01" name="total_charges" required>

        <label>Number of Support Calls</label>
        <input type="number" name="num_support_calls" min="0" required>

        <label>Senior Citizen</label>
        <select name="senior_citizen">
            <option value="0">No</option>
            <option value="1">Yes</option>
        </select>

        <label>Contract Type</label>
        <select name="contract_type">
            <option value="Month-to-month">Month-to-month</option>
            <option value="One Year">One Year</option>
            <option value="Two Year">Two Year</option>
        </select>

        <label>Internet Service</label>
        <select name="internet_service">
            <option value="DSL">DSL</option>
            <option value="Fiber Optic">Fiber Optic</option>
            <option value="No">No</option>
        </select>

        <label>Tech Support</label>
        <select name="tech_support">
            <option value="No">No</option>
            <option value="Yes">Yes</option>
        </select>

        <label>Online Security</label>
        <select name="online_security">
            <option value="No">No</option>
            <option value="Yes">Yes</option>
        </select>

        <label>Payment Method</label>
        <select name="payment_method">
            <option value="Electronic Check">Electronic Check</option>
            <option value="Mailed Check">Mailed Check</option>
            <option value="Bank Transfer">Bank Transfer</option>
            <option value="Credit Card">Credit Card</option>
        </select>

        <label>Partner</label>
        <select name="partner">
            <option value="No">No</option>
            <option value="Yes">Yes</option>
        </select>

        <label>Paperless Billing</label>
        <select name="paperless_billing">
            <option value="No">No</option>
            <option value="Yes">Yes</option>
        </select>

        <button type="submit">
            Predict Customer Churn
        </button>

    </form>

    {% if result %}

    <div class="result">

        <h2>{{ result }}</h2>

        <p>
            Churn Probability:
            <strong>{{ probability }}%</strong>
        </p>

        {% if probability|float >= 70 %}
            <p class="risk">High Risk Customer</p>
        {% elif probability|float >= 40 %}
            <p class="risk">Medium Risk Customer</p>
        {% else %}
            <p class="risk">Low Risk Customer</p>
        {% endif %}

    </div>

    {% endif %}

</div>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    probability = None

    if request.method == "POST":

        data = {
            "tenure_months": float(request.form["tenure_months"]),
            "monthly_charges": float(request.form["monthly_charges"]),
            "total_charges": float(request.form["total_charges"]),
            "num_support_calls": int(request.form["num_support_calls"]),
            "senior_citizen": int(request.form["senior_citizen"]),

            "contract_type": request.form["contract_type"],
            "internet_service": request.form["internet_service"],
            "tech_support": request.form["tech_support"],
            "online_security": request.form["online_security"],
            "payment_method": request.form["payment_method"],
            "partner": request.form["partner"],
            "paperless_billing": request.form["paperless_billing"]
        }

        # Convert input into DataFrame
        input_data = pd.DataFrame([data])

        # Prediction
        prediction = pipeline["model"].predict(
            pipeline["preprocessor"].transform(input_data)
        )[0]

        probability = pipeline["model"].predict_proba(
            pipeline["preprocessor"].transform(input_data)
        )[0][1] * 100

        probability = round(probability, 2)

        if prediction == 1:
            result = "Customer is likely to CHURN"
        else:
            result = "Customer is likely to STAY"

    return render_template_string(
        HTML,
        result=result,
        probability=probability
    )


if __name__ == "__main__":
    app.run(debug=True)