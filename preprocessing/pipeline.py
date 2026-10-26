"""W2D5: end-to-end preprocessing pipeline for the loan approval dataset."""
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "loan_approval.csv"
SCALER_PATH = ROOT / "models" / "scaler.pkl"

# Fill values learned from the full dataset in W2D2 (median for numeric, mode for text).
# Fixed constants, so a single record can be preprocessed at serving time.
FILL_VALUES = {
    "LoanAmount": 128.0,
    "Loan_Amount_Term": 360.0,
    "Credit_History": 1.0,
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Self_Employed": "No",
}

# Same columns, same order as the scaler fitted in W2D4
NUM_COLS = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "TotalIncome", "LogTotalIncome", "LoanAmount_to_Income",
]

# Fixed category lists keep the one-hot columns identical for any input
CATEGORIES = {
    "Gender": ["Female", "Male"],
    "Married": ["No", "Yes"],
    "Dependents": ["0", "1", "2", "3+"],
    "Education": ["Graduate", "Not Graduate"],
    "Self_Employed": ["No", "Yes"],
    "Property_Area": ["Rural", "Semiurban", "Urban"],
}


def preprocess(df: pd.DataFrame, scaler=None):
    """Clean, engineer features, encode and scale a raw loan DataFrame.
    Uses the scaler saved in models/scaler.pkl (fitted on training data in W2D4).
    Returns X (model-ready features) and y (1 = approved, 0 = rejected; None if no target).
    """
    df = df.copy()

    # 1. Clean missing values with fixed fill values
    df = df.fillna(FILL_VALUES)

    # 2. Engineer features (same as W2D3)
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["LoanAmount_to_Income"] = df["LoanAmount"] / df["TotalIncome"]
    df["LogTotalIncome"] = np.log1p(df["TotalIncome"])

    # 3. Scale the continuous features
    if scaler is None:
        if SCALER_PATH.exists():
            scaler = joblib.load(SCALER_PATH)
        else:
            warnings.warn("models/scaler.pkl not found; fitting a new scaler on this data")
            scaler = StandardScaler().fit(df[NUM_COLS])
    scaled = pd.DataFrame(scaler.transform(df[NUM_COLS]), columns=NUM_COLS, index=df.index)

    # 4. One-hot encode categoricals (drop_first avoids redundant columns)
    cats = pd.DataFrame(index=df.index)
    for col, levels in CATEGORIES.items():
        cats[col] = pd.Categorical(df[col].astype(str), categories=levels)
    encoded = pd.get_dummies(cats, drop_first=True, dtype=int)

    X = pd.concat([scaled, df[["Credit_History"]], encoded], axis=1)
    y = (df["Loan_Status"] == "Y").astype(int) if "Loan_Status" in df.columns else None
    return X, y


if __name__ == "__main__":
    raw = pd.read_csv(RAW_CSV)
    X, y = preprocess(raw)

    assert X.isnull().sum().sum() == 0, "NaN values remain in X"
    assert len(X) == len(y), "X and y have different lengths"

    print("X shape:", X.shape)
    print("y shape:", y.shape, "| approved:", int(y.sum()), "| rejected:", int((y == 0).sum()))
    print("\nColumns:", list(X.columns))
    print("\nFirst 5 rows of X:")
    print(X.head().round(3).to_string())
