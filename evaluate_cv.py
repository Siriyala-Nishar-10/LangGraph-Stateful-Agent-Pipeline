"""W4D1: 5-fold cross-validation and learning curve for the best model."""
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save the figure to a file without opening a window
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve

from preprocessing.pipeline import RAW_CSV, preprocess

MODELS_DIR = Path(__file__).resolve().parent / "models"

# 1. Load the best model and the data; clone gives an unfitted copy for cross-validation
best = joblib.load(MODELS_DIR / "best_model.pkl")
name = type(best).__name__
X, y = preprocess(pd.read_csv(RAW_CSV))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2. 5-fold cross-validation
scores = cross_val_score(clone(best), X, y, cv=cv, scoring="accuracy")
mean, std = scores.mean(), scores.std()
print(f"Model: {name}")
print("Fold accuracies:", np.round(scores, 3))
print(f"CV accuracy: {mean:.3f} +/- {std:.3f}")

# 3. Learning curve: training vs validation score across training set sizes
sizes, train_scores, val_scores = learning_curve(
    clone(best), X, y, cv=cv, scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 8), random_state=42, shuffle=True,
)
train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)
print("\nTraining size | train score | validation score")
for n, tr, va in zip(sizes, train_mean, val_mean):
    print(f"{n:>13} | {tr:.3f}       | {va:.3f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sizes, train_mean, "o-", label="training score")
ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
ax.plot(sizes, val_mean, "o-", label="validation score (5-fold CV)")
ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)
ax.set_title(f"Learning curve: {name}")
ax.set_xlabel("training set size")
ax.set_ylabel("accuracy")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(MODELS_DIR / "learning_curve.png", dpi=150)

# 4. Append the CV scores to models/results.csv (replace rows from an earlier run)
results_path = MODELS_DIR / "results.csv"
results = pd.read_csv(results_path)
cv_name = f"{name} (5-fold CV)"
today = date.today().isoformat()
new_rows = pd.DataFrame([
    {"model_name": cv_name, "metric_name": "accuracy_mean", "score": round(mean, 4), "date": today},
    {"model_name": cv_name, "metric_name": "accuracy_std", "score": round(std, 4), "date": today},
])
results = results[results["model_name"] != cv_name]
pd.concat([results, new_rows], ignore_index=True).to_csv(results_path, index=False)
print("\nSaved models/learning_curve.png and updated models/results.csv")
print(pd.read_csv(results_path).to_string(index=False))
