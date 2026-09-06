"""
Inference pipeline for the Telco Churn model.

Loads the MLflow-logged model and reproduces the exact feature
transformations used at training time, so serving predictions match
what the model was trained on.
"""

import os
import glob
import pandas as pd
import mlflow

# Set at Docker build time. Falls back to the latest local MLflow run
# when developing outside the container.
MODEL_DIR = "/app/model"

try:
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"Model loaded from {MODEL_DIR}")
except Exception as e:
    print(f"Could not load model from {MODEL_DIR}: {e}")
    local_model_paths = glob.glob("./mlruns/*/*/artifacts/model") + glob.glob("./src/serving/model/*/artifacts/model")
    if not local_model_paths:
        raise Exception(f"No model found locally either. Original error: {e}")
    latest_model = max(local_model_paths, key=os.path.getmtime)
    model = mlflow.pyfunc.load_model(latest_model)
    MODEL_DIR = os.path.dirname(latest_model)  # parent "artifacts" dir holds feature_columns.txt
    print(f"Loaded model from local run: {latest_model}")

with open(os.path.join(MODEL_DIR, "feature_columns.txt")) as f:
    FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]

# Binary features are mapped explicitly instead of relying on one-hot
# encoding, so the same 0/1 values are produced at train and serve time.
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same encoding used during training to a raw input row."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c].astype(str).str.strip()
                .map(mapping).astype("Int64")
                .fillna(0).astype(int)
            )

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # Missing columns are filled with 0, extras are dropped, so the model
    # always receives features in the exact order it was trained on.
    return df.reindex(columns=FEATURE_COLS, fill_value=0)


def predict(input_dict: dict) -> str:
    """Predict churn for one customer, given as a dict of raw fields."""
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)

    try:
        preds = model.predict(df_enc)
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        result = preds[0] if isinstance(preds, (list, tuple)) else preds
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")

    return "Likely to churn" if result == 1 else "Not likely to churn"
