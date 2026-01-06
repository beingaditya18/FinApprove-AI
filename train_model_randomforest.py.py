"""
RandomForest Loan Model (Balanced with SMOTE)
SMOTE applied AFTER encoding categorical columns.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# ===== CONFIG =====
DATA_FILE = "loan_dataset_1000.csv"
MODEL_FILE = "model_randomforest.pkl"

print(f"[INFO] Loading dataset: {DATA_FILE}")
df = pd.read_csv(DATA_FILE)

# Fix missing target
df['Loan_Status'] = df['Loan_Status'].fillna('Rejected')

df['Loan_Status_bin'] = df['Loan_Status'].map({
    'Approved': 1,
    'Rejected': 0,
    'Conditional Approval': 1
})

# Features
features = [
    "Age","Dependents","Annual_Income","Credit_Score","Existing_Loans",
    "Loan_Amount","Loan_Term_Months","Collateral_Value","Late_Payments_Count",
    "Gender","Marital_Status","Education_Level","Employment_Type",
    "Loan_Purpose","City"
]

X = df[features]
y = df['Loan_Status_bin']

numeric_cols = [
    "Age","Dependents","Annual_Income","Credit_Score","Existing_Loans",
    "Loan_Amount","Loan_Term_Months","Collateral_Value","Late_Payments_Count"
]

categorical_cols = [
    "Gender","Marital_Status","Education_Level",
    "Employment_Type","Loan_Purpose","City"
]

# STEP 1: Encode data BEFORE SMOTE
encoder = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
])

print("[INFO] Encoding data before SMOTE...")
X_encoded = encoder.fit_transform(X)

# STEP 2: Apply SMOTE
print("[INFO] Applying SMOTE...")
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_encoded, y)

print("Before SMOTE:", y.value_counts().to_dict())
print("After SMOTE:", pd.Series(y_bal).value_counts().to_dict())

# STEP 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42
)

# STEP 4: Final model
model = RandomForestClassifier(
    n_estimators=350,
    random_state=42,
    n_jobs=-1
)

print("[TRAINING] Training Random Forest...")
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"[RESULT] Accuracy: {acc:.4f}")

# STEP 5: Save as a pipeline (encoder + model)
final_pipeline = Pipeline([
    ('preprocess', encoder),
    ('model', model)
])

joblib.dump(final_pipeline, MODEL_FILE)
print(f"[SAVED] Model saved as {MODEL_FILE}")
