"""Model loading and single-application prediction, used by the FastAPI app."""
from pathlib import Path

import joblib
import pandas as pd

from preprocessing.pipeline import preprocess

ROOT = Path(__file__).resolve().parent
NUMERIC_FIELDS = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History",
]


def load_model():
    """Load the tuned model if it exists, otherwise the best model."""
    for name in ("tuned_model.pkl", "best_model.pkl"):
        path = ROOT / "models" / name
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError("No model found in models/ (expected tuned_model.pkl or best_model.pkl)")


def predict_application(model, data: dict) -> dict:
    """Preprocess one loan application (missing fields are filled) and return the prediction."""
    df = pd.DataFrame([data])
    for col in NUMERIC_FIELDS:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # None becomes NaN, filled by the pipeline
    X, _ = preprocess(df)
    probability = float(model.predict_proba(X)[0, 1])
    approved = probability >= 0.5
    return {
        "decision": "approved" if approved else "rejected",
        "approved": bool(approved),
        "probability_approved": round(probability, 4),
        "model": type(model).__name__,
    }
