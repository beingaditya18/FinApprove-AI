"""
train_model_xgboost.py
Trains XGBoost Loan Approval Model.
Saves model as model_xgboost.pkl
"""

import os
import joblib
import pandas as pd
# Need to import XGBClassifier
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Import AdaBoostClassifier if you want to keep the same structure for imports
# from sklearn.ensemble import AdaBoostClassifier

# ====== CONFIG ======
DATA_FILE = "loan_dataset_1000.csv"
MODEL_FILE = "model_xgboost.pkl"  # Must match app.py configuration
print(f"[INFO] Loading dataset: {DATA_FILE}")

# --- Data Loading and Preparation (Identical to AdaBoost script) ---
df = pd.read_csv(DATA_FILE)

# Fill missing values
df['Loan_Status'] = df['Loan_Status'].fillna('Rejected')

# Convert Loan Status to binary
df['Loan_Status_bin'] = df['Loan_Status'].map({
    'Approved': 1,
    'Rejected': 0,
    'Conditional Approval': 1  # treat conditional as approved
})

# Feature order must match app.py
features = [
    "Age","Dependents","Annual_Income","Credit_Score","Existing_Loans","Loan_Amount",
    "Loan_Term_Months","Collateral_Value","Late_Payments_Count","Gender","Marital_Status",
    "Education_Level","Employment_Type","Loan_Purpose","City"
]

X = df[features]
y = df['Loan_Status_bin']

numeric_cols = ["Age","Dependents","Annual_Income","Credit_Score","Existing_Loans",
                "Loan_Amount","Loan_Term_Months","Collateral_Value","Late_Payments_Count"]

categorical_cols = ["Gender","Marital_Status","Education_Level","Employment_Type","Loan_Purpose","City"]

# Preprocessing pipeline (Identical)
preprocess = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
])

# --- Model Specific Configuration ---
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    use_label_encoder=False, 
    eval_metric='logloss',
    random_state=42
)

pipeline = Pipeline([
    ('preprocess', preprocess),
    ('model', model)
])

# Train Test Split (Identical)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("[TRAINING] Training XGBoost Model...")
pipeline.fit(X_train, y_train)

accuracy = pipeline.score(X_test, y_test)
print(f"[RESULT] Accuracy: {accuracy:.4f}")

joblib.dump(pipeline, MODEL_FILE)
print(f"[SAVED] Model saved as {MODEL_FILE}")