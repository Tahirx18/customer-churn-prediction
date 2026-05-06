"""
Customer Churn Prediction - Flask Web Application
"""
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import pickle
import numpy as np
import os
import random

app = Flask(__name__)

# Load model and encoders at startup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "customer_churn_model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")
DATA_PATH = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn-Updated.xlsx")

with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

loaded_model = model_data["model"]
feature_names = model_data["features_names"]

with open(ENCODERS_PATH, "rb") as f:
    encoders = pickle.load(f)

# Load dataset for dashboard analytics
try:
    df = pd.read_excel(DATA_PATH)
except Exception:
    df = None

# Define the options for each categorical field (for the frontend)
FIELD_OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No phone service", "No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}


@app.route("/")
def home():
    return redirect(url_for('login'))


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/predict-page")
def index():
    return render_template("index.html", field_options=FIELD_OPTIONS)


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_data = {
            'gender': data.get('gender', 'Female'),
            'SeniorCitizen': int(data.get('SeniorCitizen', 0)),
            'Partner': data.get('Partner', 'No'),
            'Dependents': data.get('Dependents', 'No'),
            'tenure': int(data.get('tenure', 0)),
            'PhoneService': data.get('PhoneService', 'No'),
            'MultipleLines': data.get('MultipleLines', 'No phone service'),
            'InternetService': data.get('InternetService', 'DSL'),
            'OnlineSecurity': data.get('OnlineSecurity', 'No'),
            'OnlineBackup': data.get('OnlineBackup', 'No'),
            'DeviceProtection': data.get('DeviceProtection', 'No'),
            'TechSupport': data.get('TechSupport', 'No'),
            'StreamingTV': data.get('StreamingTV', 'No'),
            'StreamingMovies': data.get('StreamingMovies', 'No'),
            'Contract': data.get('Contract', 'Month-to-month'),
            'PaperlessBilling': data.get('PaperlessBilling', 'No'),
            'PaymentMethod': data.get('PaymentMethod', 'Electronic check'),
            'MonthlyCharges': float(data.get('MonthlyCharges', 0)),
            'TotalCharges': float(data.get('TotalCharges', 0)),
        }

        input_df = pd.DataFrame([input_data])

        # Encode categorical features
        for column, encoder in encoders.items():
            if column in input_df.columns:
                input_df[column] = encoder.transform(input_df[column])

        # Predict
        prediction = loaded_model.predict(input_df)[0]
        probabilities = loaded_model.predict_proba(input_df)[0]

        churn_prob = round(float(probabilities[1]) * 100, 1)
        no_churn_prob = round(float(probabilities[0]) * 100, 1)

        # Risk level
        if churn_prob >= 70:
            risk_level = "Critical"
            risk_color = "#ff4757"
        elif churn_prob >= 40:
            risk_level = "High"
            risk_color = "#ff6b35"
        elif churn_prob >= 20:
            risk_level = "Medium"
            risk_color = "#ffa502"
        else:
            risk_level = "Low"
            risk_color = "#2ed573"

        return jsonify({
            "success": True,
            "prediction": int(prediction),
            "churn_probability": churn_prob,
            "no_churn_probability": no_churn_prob,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "label": "Churn" if prediction == 1 else "No Churn"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/model-info")
def model_info():
    return jsonify({
        "model_type": "Random Forest Classifier",
        "features": feature_names,
        "accuracy": "77.8%",
        "total_features": len(feature_names),
    })


@app.route("/api/dashboard-data")
def dashboard_data():
    """Returns all data needed for the analytics dashboard."""
    if df is None:
        return jsonify({"error": "Dataset not found"}), 404

    total_customers = len(df)

    # Determine churn column name
    churn_col = None
    for col in ['Churn', 'churn', 'CHURN']:
        if col in df.columns:
            churn_col = col
            break

    if churn_col is None:
        return jsonify({"error": "Churn column not found"}), 404

    churn_values = df[churn_col].astype(str).str.strip().str.lower()
    churners = int((churn_values.isin(['yes', '1', 'true'])).sum())
    retained = total_customers - churners
    retention_rate = round((retained / total_customers) * 100, 1) if total_customers > 0 else 0
    churn_rate = round((churners / total_customers) * 100, 1) if total_customers > 0 else 0

    # Monthly charges distribution for chart
    if 'MonthlyCharges' in df.columns:
        avg_monthly = round(float(df['MonthlyCharges'].mean()), 2)
    else:
        avg_monthly = 0

    # Generate churn probability distribution (simulated from model on sample)
    distribution_bins = []
    try:
        sample = df.sample(min(500, len(df)), random_state=42).copy()
        sample_encoded = sample.copy()
        for column, encoder in encoders.items():
            if column in sample_encoded.columns:
                try:
                    sample_encoded[column] = encoder.transform(sample_encoded[column])
                except Exception:
                    pass
        
        # Keep only feature columns
        feature_cols = [f for f in feature_names if f in sample_encoded.columns]
        if feature_cols:
            sample_features = sample_encoded[feature_cols]
            probas = loaded_model.predict_proba(sample_features)[:, 1] * 100
            
            bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
            hist, _ = np.histogram(probas, bins=bins)
            distribution_bins = [{"range": labels[i], "count": int(hist[i])} for i in range(len(labels))]
    except Exception:
        # Fallback simulated distribution
        distribution_bins = [
            {"range": "0-10", "count": 180},
            {"range": "10-20", "count": 95},
            {"range": "20-30", "count": 62},
            {"range": "30-40", "count": 45},
            {"range": "40-50", "count": 38},
            {"range": "50-60", "count": 28},
            {"range": "60-70", "count": 22},
            {"range": "70-80", "count": 15},
            {"range": "80-90", "count": 10},
            {"range": "90-100", "count": 5},
        ]

    # Generate recent customers table data from actual dataset
    recent_customers = []
    try:
        sample_rows = df.sample(min(12, len(df)), random_state=55).copy()
        sample_encoded_rows = sample_rows.copy()
        for column, encoder in encoders.items():
            if column in sample_encoded_rows.columns:
                try:
                    sample_encoded_rows[column] = encoder.transform(sample_encoded_rows[column])
                except Exception:
                    pass
        
        feature_cols = [f for f in feature_names if f in sample_encoded_rows.columns]
        if feature_cols:
            probas = loaded_model.predict_proba(sample_encoded_rows[feature_cols])[:, 1] * 100
            
            first_names = ["James", "Maria", "Robert", "Linda", "David", "Sarah",
                           "Michael", "Emily", "William", "Jessica", "Daniel", "Ashley"]
            last_names = ["Thompson", "Garcia", "Anderson", "Martinez", "Johnson", "Wilson",
                          "Taylor", "Lee", "Harris", "Brown", "Clark", "Lewis"]
            
            for i, (idx, row) in enumerate(sample_rows.iterrows()):
                prob = round(float(probas[i]), 1)
                if prob >= 60:
                    risk = "High"
                elif prob >= 30:
                    risk = "Medium"
                else:
                    risk = "Low"
                
                name = f"{first_names[i % len(first_names)]} {last_names[i % len(last_names)]}"
                tenure_val = int(row.get('tenure', 0))
                monthly = round(float(row.get('MonthlyCharges', 0)), 2)
                contract = str(row.get('Contract', 'N/A'))
                
                recent_customers.append({
                    "id": f"CUST-{1000 + i}",
                    "name": name,
                    "tenure": tenure_val,
                    "monthlyCharges": monthly,
                    "contract": contract,
                    "churnProbability": prob,
                    "riskLevel": risk
                })
    except Exception:
        pass

    # Overall churn risk indicator
    if churn_rate >= 30:
        overall_risk = "High"
    elif churn_rate >= 15:
        overall_risk = "Elevated"
    else:
        overall_risk = "Stable"

    return jsonify({
        "kpis": {
            "totalCustomers": total_customers,
            "predictedChurners": churners,
            "retentionRate": retention_rate,
            "avgMonthlyCharges": avg_monthly,
            "churnRate": churn_rate
        },
        "overallRisk": overall_risk,
        "churnDistribution": distribution_bins,
        "recentCustomers": recent_customers
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
