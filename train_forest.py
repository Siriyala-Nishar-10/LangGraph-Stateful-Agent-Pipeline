"""W3D2: train a random forest classifier and compare it with the baseline."""
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from preprocessing.pipeline import RAW_CSV, preprocess

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODEL_NAME = "RandomForestClassifier"

# 1. Same data and same 80/20 stratified split as the baseline
raw = pd.read_csv(RAW_CSV)
X, y = preprocess(raw)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Train the random forest
model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
pred = model.predict(X_test)
accuracy = round(accuracy_score(y_test, pred), 4)

# 3. Compare against the baseline stored in models/results.csv
results_path = MODELS_DIR / "results.csv"
results = pd.read_csv(results_path)
baseline = results.loc[results["model_name"] == "LogisticRegression", "score"].iloc[0]
print(f"Baseline (LogisticRegression) accuracy: {baseline:.3f}")
print(f"{MODEL_NAME} accuracy:            {accuracy:.3f}")
print(f"Difference: {accuracy - baseline:+.3f}")

# 4. Top 5 most important features
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 5 features by importance:")
print(importances.head(5).round(3).to_string())

# 5. Save scored records (test rows) with the predicted probability of approval
scored = pd.DataFrame({
    "Loan_ID": raw.loc[y_test.index, "Loan_ID"],
    "actual": y_test,
    "predicted": pred,
    "approval_probability": model.predict_proba(X_test)[:, 1].round(4),
})
scored.to_csv(ROOT / "data" / "scored_records.csv", index=False)

# 6. Append the new score to models/results.csv (replace an earlier run of the same model)
row = pd.DataFrame([{
    "model_name": MODEL_NAME,
    "metric_name": "accuracy",
    "score": accuracy,
    "date": date.today().isoformat(),
}])
results = results[~((results["model_name"] == MODEL_NAME) & (results["metric_name"] == "accuracy"))]
pd.concat([results, row], ignore_index=True).to_csv(results_path, index=False)
print("\nSaved data/scored_records.csv and updated models/results.csv")
