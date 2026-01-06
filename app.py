import os
import time
import joblib
import traceback
import pandas as pd
from flask import Flask, request, jsonify, send_file, render_template

app = Flask(__name__, template_folder="templates")
app.config['TEMPLATES_AUTO_RELOAD'] = True

# --- 1. Model Configuration and Loading ---
BASE = os.path.dirname(__file__)

MODEL_FILENAMES = {
    'adaboost': "model_adaboost.pkl",
    'randomforest': "model_randomforest.pkl",
    'xgboost': "model_xgboost.pkl",
}

LOADED_MODELS = {}

def load_models():
    app.logger.info("Attempting to load ML models...")
    for key, filename in MODEL_FILENAMES.items():
        model_path = os.path.join(BASE, filename)
        try:
            model = joblib.load(model_path)
            LOADED_MODELS[key] = model
            app.logger.info(f"Loaded {key.upper()} model from {filename}")
        except FileNotFoundError:
            app.logger.error(f"ERROR: Model file {filename} not found at {model_path}.")
        except Exception as e:
            app.logger.exception(f"Failed to load {filename}: {e}")

load_models()
# ----------------------------------------------------


# ===========================
# FRONTEND ROUTE
# ===========================
@app.route('/')
def home():
    return render_template("index.html")
# ===========================


def prepare_input(data):
    return {
        "Age": int(data.get("Age", 32)),
        "Gender": data.get("Gender", "Male"),
        "Marital_Status": data.get("Marital_Status", "Single"),
        "Dependents": int(data.get("Dependents", 0)),
        "Education_Level": data.get("Education_Level", "Graduate"),
        "Employment_Type": data.get("Employment_Type", "Salaried"),
        "Annual_Income": float(data.get("Annual_Income", 600000)),
        "Credit_Score": float(data.get("Credit_Score", 720)),
        "Existing_Loans": int(data.get("Existing_Loans", 0)),
        "Loan_Amount": float(data.get("Loan_Amount", 250000)),
        "Loan_Term_Months": int(data.get("Loan_Term_Months", 60)),
        "Collateral_Value": float(data.get("Collateral_Value", 0)),
        "Loan_Purpose": data.get("Loan_Purpose", "Personal"),
        "Late_Payments_Count": int(data.get("Late_Payments_Count", 0)),
        "City": data.get("City", "Indore")
    }


def make_dataframe(fields):
    cols = [
        "Age","Dependents","Annual_Income","Credit_Score","Existing_Loans","Loan_Amount",
        "Loan_Term_Months","Collateral_Value","Late_Payments_Count",
        "Gender","Marital_Status","Education_Level","Employment_Type",
        "Loan_Purpose","City"
    ]

    row = [[
        fields["Age"], fields["Dependents"], fields["Annual_Income"], fields["Credit_Score"],
        fields["Existing_Loans"], fields["Loan_Amount"], fields["Loan_Term_Months"],
        fields["Collateral_Value"], fields["Late_Payments_Count"],
        fields["Gender"], fields["Marital_Status"], fields["Education_Level"],
        fields["Employment_Type"], fields["Loan_Purpose"], fields["City"]
    ]]

    return pd.DataFrame(row, columns=cols)


def explain(pred, fields):
    reasons = []
    
    if fields["Credit_Score"] >= 750:
        reasons.append("Excellent credit profile (Credit Score: 750+)")
    elif fields["Credit_Score"] >= 670:
        reasons.append("Good credit profile (Credit Score: 670+)")
    elif fields["Credit_Score"] >= 580:
        reasons.append("Fair credit profile (Credit Score: 580+)")
    else:
        reasons.append("Low credit score (below 580)")

    lti = fields["Loan_Amount"] / fields["Annual_Income"] if fields["Annual_Income"] > 0 else 999
    reasons.append("Loan-to-income ratio is favourable" if lti < 0.8 else "High loan-to-income ratio")

    reasons.append("Multiple late payments" if fields["Late_Payments_Count"] > 2 else "Few or no late payments")

    monthly_income = fields["Annual_Income"] / 12.0 if fields["Annual_Income"] > 0 else 1.0
    emi_est = fields["Loan_Amount"] / max(fields["Loan_Term_Months"], 1)
    existing_loan_cost = fields["Existing_Loans"] * 5000
    dti = (existing_loan_cost + emi_est) / monthly_income if monthly_income > 0 else 999
    reasons.append("Acceptable DTI ratio" if dti < 0.5 else "High DTI ratio")

    if pred == 1:
        approved_r = [r for r in reasons if "High" not in r and "Low" not in r]
        return "Approved: " + "; ".join(approved_r) if approved_r else "Approved."
    else:
        rejected_r = [r for r in reasons if "High" in r or "Low" in r]
        return "Rejected: " + "; ".join(rejected_r) if rejected_r else "Rejected."


@app.route('/predict', methods=['POST'])
def predict():
    model_key = request.args.get('model', 'adaboost').lower()
    
    try:
        if model_key not in LOADED_MODELS:
            return jsonify({"status": "Error", "message": f"Model '{model_key}' not available."}), 400

        current_model = LOADED_MODELS[model_key]

        data = request.form.to_dict()
        fields = prepare_input(data)
        X_df = make_dataframe(fields)

        start = time.time()
        pred = int(current_model.predict(X_df)[0])

        # -------- FIX RANDOM FOREST WRONG CLASS ORDER ----------
        if model_key == "randomforest":
            try:
                class_order = current_model.classes_.tolist()
                if class_order == [1, 0]:    # inverted class order
                    pred = 1 - pred
            except:
                pass
        # -------------------------------------------------------

        prob = None
        if hasattr(current_model, "predict_proba"):
            if pred in current_model.classes_:
                prob_index = current_model.classes_.tolist().index(pred)
                prob = float(current_model.predict_proba(X_df)[0][prob_index])
            else:
                prob = float(max(current_model.predict_proba(X_df)[0]))

        elapsed = round((time.time() - start) * 1000, 3)

        status = "Approved" if pred == 1 else "Rejected"
        reason = explain(pred, fields)

        return jsonify({
            "model": model_key,
            "status": status,
            "probability": prob,
            "reason": reason,
            "prediction_time_ms": elapsed
        })
        
    except Exception as e:
        app.logger.exception(f"Prediction failed for model {model_key}: {e}")
        return jsonify({"status": "Error", "message": str(e)}), 500


@app.route('/download_model')
def download_model():
    model_key = request.args.get('model', 'adaboost').lower()
    filename = MODEL_FILENAMES.get(model_key)

    if not filename:
        return "Model key not recognized", 400
        
    model_path = os.path.join(BASE, filename)
    if not os.path.exists(model_path):
        return f"Model file '{filename}' not found", 404
        
    return send_file(model_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
