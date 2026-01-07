import os
import time
import joblib
import traceback
import pandas as pd
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Load only AdaBoost model
BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "model_adaboost.pkl")

try:
    model = joblib.load(MODEL_PATH)
    app.logger.info(f"Loaded AdaBoost model from {MODEL_PATH}")
except FileNotFoundError:
    model = None
    app.logger.error(f"ERROR: model_adaboost.pkl not found at {MODEL_PATH}")
except Exception as e:
    model = None
    app.logger.exception(f"Failed to load model_adaboost.pkl: {e}")

@app.route('/')
def home():
    return "✅ Loan Prediction API is running!"

def prepare_input(data):
    return {
        "Age": int(data.get("Age", 30)),
        "Gender": data.get("Gender", "Male"),
        "Marital_Status": data.get("Marital_Status", "Single"),
        "Dependents": int(data.get("Dependents", 0)),
        "Education_Level": data.get("Education_Level", "Graduate"),
        "Employment_Type": data.get("Employment_Type", "Salaried"),
        "Annual_Income": float(data.get("Annual_Income", 500000)),
        "Credit_Score": float(data.get("Credit_Score", 650)),
        "Existing_Loans": int(data.get("Existing_Loans", 0)),
        "Loan_Amount": float(data.get("Loan_Amount", 200000)),
        "Loan_Term_Months": int(data.get("Loan_Term_Months", 36)),
        "Collateral_Value": float(data.get("Collateral_Value", 0)),
        "Loan_Purpose": data.get("Loan_Purpose", "Personal"),
        "Late_Payments_Count": int(data.get("Late_Payments_Count", 0)),
        "City": data.get("City", "Indore")
    }

def make_dataframe(fields):
    cols = ["Age","Dependents","Annual_Income","Credit_Score","Existing_Loans","Loan_Amount",
            "Loan_Term_Months","Collateral_Value","Late_Payments_Count","Gender","Marital_Status",
            "Education_Level","Employment_Type","Loan_Purpose","City"]
    row = [[
        fields["Age"], fields["Dependents"], fields["Annual_Income"], fields["Credit_Score"],
        fields["Existing_Loans"], fields["Loan_Amount"], fields["Loan_Term_Months"],
        fields["Collateral_Value"], fields["Late_Payments_Count"], fields["Gender"],
        fields["Marital_Status"], fields["Education_Level"], fields["Employment_Type"],
        fields["Loan_Purpose"], fields["City"]
    ]]
    return pd.DataFrame(row, columns=cols)

def explain(pred, fields):
    reasons = []

    if fields["Credit_Score"] >= 750:
        reasons.append("Excellent credit score")
    elif fields["Credit_Score"] >= 670:
        reasons.append("Good credit score")
    elif fields["Credit_Score"] >= 580:
        reasons.append("Fair credit score")
    else:
        reasons.append("Low credit score")

    lti = fields["Loan_Amount"] / fields["Annual_Income"] if fields["Annual_Income"]>0 else 999
    reasons.append("Loan-to-income ratio is favourable" if lti < 0.8 else "High loan-to-income ratio")

    reasons.append("Multiple late payments on record" if fields["Late_Payments_Count"] > 2 else "No or few late payments")

    monthly_income = fields["Annual_Income"] / 12.0 if fields["Annual_Income"]>0 else 1
    emi_est = fields["Loan_Amount"] / max(fields["Loan_Term_Months"],1)
    dti = (fields["Existing_Loans"] * 5000 + emi_est) / monthly_income
    reasons.append("Debt-to-income ratio is acceptable" if dti < 0.5 else "High debt-to-income ratio")

    if pred == 1:
        text = "Approved because: " + "; ".join([r for r in reasons if "High" not in r and "Low" not in r])
    else:
        text = "Rejected because: " + "; ".join([r for r in reasons if "High" in r or "Low" in r])
    return text

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({"status": "Error", "message": "Model not loaded on server"}), 500

        data = request.form.to_dict()
        fields = prepare_input(data)
        X_df = make_dataframe(fields)

        start = time.time()
        pred = int(model.predict(X_df)[0])
        prob = float(max(model.predict_proba(X_df)[0])) if hasattr(model, "predict_proba") else None
        elapsed = round((time.time() - start) * 1000, 3)

        status = "Approved" if pred == 1 else "Rejected"
        reason = explain(pred, fields)

        return jsonify({
            "model": "adaboost",
            "status": status,
            "probability": prob,
            "reason": reason,
            "prediction_time_ms": elapsed
        })
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500

@app.route('/download_model')
def download_model():
    if not MODEL_PATH or not os.path.exists(MODEL_PATH):
        return "Model file not found", 404
    return send_file(MODEL_PATH, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
