"""
train_model_adaboost.py
Trains AdaBoost Loan Approval Model using sklearn 1.7.0
Saves model as model_adaboost.pkl (no change needed in app.py)
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier

# ====== CONFIG ======
DATA_FILE = "loan_dataset_1000.csv"      # Change if dataset name is different
MODEL_FILE = "model_adaboost.pkl"        # Must match app.py
print(f"[INFO] Loading dataset: {DATA_FILE}")

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

# Preprocessing pipeline
preprocess = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
])

model = AdaBoostClassifier(n_estimators=120, learning_rate=0.9, random_state=42)

pipeline = Pipeline([
    ('preprocess', preprocess),
    ('model', model)
])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("[TRAINING] Training AdaBoost Model...")
pipeline.fit(X_train, y_train)

accuracy = pipeline.score(X_test, y_test)
print(f"[RESULT] Accuracy: {accuracy:.4f}")

joblib.dump(pipeline, MODEL_FILE)
print(f"[SAVED] Model saved as {MODEL_FILE}")
