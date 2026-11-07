"""W3D1: train a logistic regression baseline on the preprocessed loan data."""
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from preprocessing.pipeline import RAW_CSV, preprocess

MODELS_DIR = Path(__file__).resolve().parent / "models"

# 1. Data from the W2D5 pipeline (same 80/20 stratified split as the W2D4 scaler)
X, y = preprocess(pd.read_csv(RAW_CSV))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Baseline model
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

# Context: accuracy of always predicting the most common class
majority = max(y_test.mean(), 1 - y_test.mean())
print(f"Train rows: {len(X_train)} | test rows: {len(X_test)}")
print(f"Baseline accuracy (LogisticRegression): {accuracy:.3f}")
print(f"Majority-class accuracy (always 'approved'): {majority:.3f}")

# 3. Predictions vs actuals
results = pd.DataFrame({"actual": y_test, "predicted": pred}, index=y_test.index)
print("\n5-row sample of predictions vs actuals:")
print(results.head(5).to_string())

# 4. Save predictions and the baseline score
MODELS_DIR.mkdir(exist_ok=True)
results.to_csv(MODELS_DIR / "linear_predictions.csv", index_label="row_id")

row = pd.DataFrame([{
    "model_name": "LogisticRegression",
    "metric_name": "accuracy",
    "score": round(accuracy, 4),
    "date": date.today().isoformat(),
}])
results_path = MODELS_DIR / "results.csv"
if results_path.exists():
    old = pd.read_csv(results_path)
    # replace an earlier run of the same model and metric instead of duplicating it
    old = old[~((old["model_name"] == "LogisticRegression") & (old["metric_name"] == "accuracy"))]
    row = pd.concat([old, row], ignore_index=True)
row.to_csv(results_path, index=False)
print("\nSaved models/linear_predictions.csv and models/results.csv")
