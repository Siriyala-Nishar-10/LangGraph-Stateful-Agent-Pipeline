"""W3D4: pick the best model from results.csv, retrain it on all data, save it and a model card."""
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from preprocessing.pipeline import RAW_CSV, preprocess

MODELS_DIR = Path(__file__).resolve().parent / "models"

MODEL_FACTORY = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000),
    "DecisionTreeClassifier": lambda: DecisionTreeClassifier(max_depth=5, random_state=42),
    "RandomForestClassifier": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
}
# If scores tie, prefer the simpler, more interpretable model
SIMPLICITY_ORDER = ["LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier"]

# 1. Identify the best model by accuracy
results = pd.read_csv(MODELS_DIR / "results.csv")
acc = results[results["metric_name"] == "accuracy"]
top = acc[acc["score"] == acc["score"].max()]
best_name = min(top["model_name"], key=SIMPLICITY_ORDER.index)
best_score = float(top["score"].iloc[0])
tied = [m for m in top["model_name"] if m != best_name]
print(results.to_string(index=False))
print(f"\nBest model: {best_name} (held-out accuracy {best_score:.4f})")
if tied:
    print(f"Tied with: {', '.join(tied)}; chose the simpler model")

# 2. Retrain on all rows (no held-out data remains after this step)
raw = pd.read_csv(RAW_CSV)
X, y = preprocess(raw)
model = MODEL_FACTORY[best_name]().fit(X, y)

# 3. Save the model
joblib.dump(model, MODELS_DIR / "best_model.pkl")
print(f"Retrained on {len(X)} rows, {X.shape[1]} features -> saved models/best_model.pkl")

# 4. Write the model card
approved = int(y.sum())
tie_note = (
    f" It tied with {', '.join(tied)} on held-out accuracy; the simpler, more interpretable model was chosen."
    if tied else ""
)
card = f"""# Model Card: {best_name}

Date: {date.today().isoformat()}
Project: LangGraph Stateful Agent Pipeline (Cynaris Solutions AI/ML internship)

## Model
- **Model:** {best_name} (scikit-learn), saved as `models/best_model.pkl` with joblib.
- **Selected by:** highest accuracy in `models/results.csv`.{tie_note}
- **Retraining:** after selection, the model was retrained on all {len(X)} rows.

## Training data
- Kaggle "Loan Prediction Problem" dataset (`data/loan_approval.csv`), {len(raw)} rows, {raw.shape[1]} columns.
- Target: `Loan_Status` (1 = approved: {approved} rows, 0 = rejected: {len(y) - approved} rows).
- Features ({X.shape[1]}): scaled income and loan columns, engineered ratios (`TotalIncome`,
  `LoanAmount_to_Income`, `LogTotalIncome`), `Credit_History`, and one-hot encoded categories.
  Produced by `preprocessing/pipeline.py`.

## Metric
- **Held-out accuracy: {best_score:.4f}** on 123 test rows (20% stratified split, `random_state=42`),
  measured before the final retraining.
- For comparison, always predicting "approved" scores about 0.69.

## Known limitations
- **No held-out score for the saved model.** It was retrained on all rows, so its accuracy cannot be
  checked on unseen data; the figure above comes from the earlier model trained on 80% of the rows.
- **Small test set.** 123 rows means the accuracy has roughly a plus or minus 6 point margin, so
  models within a point or two of each other are not meaningfully different.
- **Accuracy only.** Precision, recall and AUC have not been evaluated yet; with a 69/31 class split
  they matter, especially the rate of wrongly approved loans.
- **Dominated by one feature.** `Credit_History` carries most of the predictive signal; income features add little.
- **Possible bias.** Features include `Gender`, `Married` and `Dependents`. A model like this should not
  be used for real lending decisions without a fairness review.
- **Data origin.** Public Kaggle data of unknown provenance, small and possibly unrepresentative.
  Missing-value fill constants were computed on the full dataset, which leaks a small amount of
  information from the test rows.
- **Not part of the agent logic yet.** This is a standalone classifier for the Week 3 tasks.
"""
(MODELS_DIR / "model_card.md").write_text(card, encoding="utf-8")
print("Saved models/model_card.md")
