"""W4D3: compute classification metrics for the tuned model and write docs/model_performance.md."""
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save the figure to a file without opening a window
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from preprocessing.pipeline import RAW_CSV, preprocess

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)

# 1. Same split as before. tuned_model.pkl was retrained on all rows, so its predictions on the
#    test rows would be contaminated; refit the same settings on the training rows only.
tuned = joblib.load(ROOT / "models" / "tuned_model.pkl")
name = type(tuned).__name__
X, y = preprocess(pd.read_csv(RAW_CSV))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = clone(tuned).fit(X_train, y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]


def report(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (approved)": precision_score(y_true, y_pred),
        "Recall (approved)": recall_score(y_true, y_pred),
        "F1 (approved)": f1_score(y_true, y_pred),
        "Specificity (rejected caught)": tn / (tn + fp),
        "ROC AUC": roc_auc_score(y_true, y_prob),
        "_cm": (int(tn), int(fp), int(fn), int(tp)),
    }


# 2. Metrics on the 123 held-out rows, and on all 614 rows via 5-fold cross-validated predictions
held = report(y_test, pred, prob)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_prob = cross_val_predict(clone(tuned), X, y, cv=cv, method="predict_proba")[:, 1]
cvr = report(y, (cv_prob >= 0.5).astype(int), cv_prob)

metric_names = [k for k in held if not k.startswith("_")]
table = "\n".join(
    f"| {m} | {held[m]:.3f} | {cvr[m]:.3f} |" for m in metric_names
)
for m in metric_names:
    print(f"{m:32s} held-out {held[m]:.3f} | 5-fold CV {cvr[m]:.3f}")

# 3. Confusion matrix heatmap (held-out rows)
tn, fp, fn, tp = held["_cm"]
cm = np.array([[tn, fp], [fn, tp]])
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1], labels=["Rejected", "Approved"])
ax.set_yticks([0, 1], labels=["Rejected", "Approved"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion matrix: {name} (123 test rows)")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(DOCS_DIR / "confusion_matrix.png", dpi=150)

# 4. Write docs/model_performance.md
wrong_approved = fp / (fp + tn)
doc = f"""# Model Performance: {name} (tuned)

Date: {date.today().isoformat()}
Task: predict whether a loan application is approved (1) or rejected (0).

## How the numbers were produced
- **Held-out column:** the tuned settings refitted on the 491 training rows and scored on the 123 test rows
  (20% stratified split, `random_state=42`). `models/tuned_model.pkl` itself is retrained on all rows, so it
  cannot be scored on unseen data.
- **5-fold CV column:** predictions for all 614 rows, each made by a model that did not see that row.
  This is the more reliable figure because it uses five times as many rows.
- "Approved" is the positive class. Decision threshold 0.5.

## Metrics

| Metric | Held-out (123 rows) | 5-fold CV (614 rows) |
|--------|---------------------|----------------------|
{table}

![Confusion matrix](confusion_matrix.png)

Held-out confusion matrix: {tp} approved loans correctly approved, {tn} rejected loans correctly rejected,
{fp} rejected loans wrongly approved, {fn} approved loans wrongly rejected.

## Which metrics matter here
- **Accuracy is not enough.** The classes are imbalanced (about 69% approved), so always predicting
  "approved" already scores about 0.69.
- **Precision (approved)** answers: of the loans the model approves, how many should have been approved?
  A wrong approval is the costly mistake in lending.
- **Recall (approved)** answers: of the loans that should be approved, how many does the model approve?
  A wrong rejection loses a good customer.
- **Specificity** is the share of bad (rejected) applications the model catches, the other side of the same trade-off.
- **F1** balances precision and recall for the approved class.
- **ROC AUC** measures how well the model ranks applications, independent of the 0.5 threshold.
- RMSE and MAE do not apply, because this is classification.

## Interpretation
- The model leans heavily towards approving. Recall for approved loans is {held["Recall (approved)"]:.2f}
  on the test rows, but specificity is only {held["Specificity (rejected caught)"]:.2f}: {fp} of the {fp + tn} rejected loans
  ({wrong_approved:.0%}) were wrongly approved. Over 5-fold CV the share of bad loans caught is
  {cvr["Specificity (rejected caught)"]:.2f}. Accuracy ({held["Accuracy"]:.3f}) hides this.
- Precision is {held["Precision (approved)"]:.2f} on the test rows ({cvr["Precision (approved)"]:.2f} in CV), so between one in six
  and one in five approvals is wrong.
- The held-out numbers are higher than the cross-validated ones (accuracy {held["Accuracy"]:.3f} against
  {cvr["Accuracy"]:.3f}, AUC {held["ROC AUC"]:.3f} against {cvr["ROC AUC"]:.3f}). With only 123 test rows the held-out
  figures are noisy, so rely on the CV column.

## Metric to optimise
**Precision for the approved class**, while keeping recall from collapsing (monitor F1).
A wrongly approved loan is a direct financial loss, whereas a wrongly rejected one is a missed opportunity,
and in this project a rejected or borderline case can also be sent to a human reviewer (the
human-in-the-loop enhancement). ROC AUC is used to compare models because it does not depend on the threshold.

The models so far were tuned for accuracy, which rewards approving almost everything on imbalanced data.
Next steps: raise the decision threshold above 0.5 or use `class_weight="balanced"`, and re-tune for
precision, F1 or AUC instead of accuracy.

## Caveats
- Test set of 123 rows: one loan changes precision or recall by about one percentage point or more.
- Features include `Gender` and `Married`; a fairness review is needed before any real use.
"""
(DOCS_DIR / "model_performance.md").write_text(doc, encoding="utf-8")
print("\nSaved docs/model_performance.md and docs/confusion_matrix.png")
