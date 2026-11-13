"""W4D2: tune the best model with RandomizedSearchCV and update the model card."""
import re
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split

from preprocessing.pipeline import RAW_CSV, preprocess

MODELS_DIR = Path(__file__).resolve().parent / "models"

# Parameter grids: 4 parameters with 2-3 values each (36 combinations for LogisticRegression)
# The L2 penalty (default) is kept: `penalty` is deprecated in recent scikit-learn versions.
PARAM_GRIDS = {
    "LogisticRegression": {
        "C": [0.1, 1, 10],
        "class_weight": [None, "balanced"],
        "solver": ["lbfgs", "liblinear", "saga"],
        "tol": [1e-4, 1e-3],
    },
    "RandomForestClassifier": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10],
        "min_samples_leaf": [1, 3, 5],
        "max_features": ["sqrt", 0.5],
    },
    "DecisionTreeClassifier": {
        "max_depth": [3, 5, 7],
        "min_samples_leaf": [1, 5, 10],
        "min_samples_split": [2, 10],
        "criterion": ["gini", "entropy"],
    },
}

best = joblib.load(MODELS_DIR / "best_model.pkl")
name = type(best).__name__
grid = PARAM_GRIDS[name]

# 1. Same 80/20 split as before: search on the training part only
X, y = preprocess(pd.read_csv(RAW_CSV))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2. Randomised search, 20 combinations x 5 folds
default = clone(best)
if name == "LogisticRegression":
    default.set_params(max_iter=5000)  # same iteration budget as the tuned candidates
search = RandomizedSearchCV(
    clone(default), grid, n_iter=20, cv=cv, scoring="accuracy", random_state=42, n_jobs=-1
)
search.fit(X_train, y_train)
print(f"Model: {name}")
print("Best parameters:", search.best_params_)
print(f"Best CV accuracy (training split): {search.best_score_:.4f}")

# 3. Compare with the default settings, same folds and same split
default_cv = cross_val_score(clone(default), X_train, y_train, cv=cv, scoring="accuracy").mean()
default_test = accuracy_score(y_test, clone(default).fit(X_train, y_train).predict(X_test))
tuned_test = accuracy_score(y_test, search.best_estimator_.predict(X_test))
print(f"\nDefault settings: CV {default_cv:.4f} | held-out test {default_test:.4f}")
print(f"Tuned settings:   CV {search.best_score_:.4f} | held-out test {tuned_test:.4f}")

# 4. Retrain on all rows with the best parameters and save
tuned = clone(default).set_params(**search.best_params_).fit(X, y)
joblib.dump(tuned, MODELS_DIR / "tuned_model.pkl")
print("Retrained on all rows -> saved models/tuned_model.pkl")

# 5. Append the tuned held-out score to models/results.csv
results_path = MODELS_DIR / "results.csv"
results = pd.read_csv(results_path)
tuned_name = f"{name} (tuned)"
row = pd.DataFrame([{
    "model_name": tuned_name, "metric_name": "accuracy",
    "score": round(tuned_test, 4), "date": date.today().isoformat(),
}])
results = results[results["model_name"] != tuned_name]
pd.concat([results, row], ignore_index=True).to_csv(results_path, index=False)

# 6. Update the model card (replace the section if this script is run again)
diff_cv, diff_test = search.best_score_ - default_cv, tuned_test - default_test
if abs(diff_cv) < 0.01 and abs(diff_test) < 0.02:
    verdict = ("The differences are within the noise of a 491-row cross-validation and a 123-row test set, "
               "so tuning gives no reliable improvement over the default settings.")
elif diff_cv > 0 and diff_test > 0:
    verdict = "Tuning improved both the cross-validated and the held-out score."
else:
    verdict = "Tuning did not improve both scores consistently, so the default settings remain a reasonable choice."

section = f"""## Tuning (W4D2)
Date: {date.today().isoformat()}
- **Search:** RandomizedSearchCV, 20 random combinations, 5-fold stratified CV, on the 491 training rows.
- **Grid:** {", ".join(f"`{k}` {v}" for k, v in grid.items())}
- **Best parameters:** `{search.best_params_}`
- **CV accuracy (training split):** default {default_cv:.4f} -> tuned {search.best_score_:.4f} ({diff_cv:+.4f})
- **Held-out test accuracy (123 rows):** default {default_test:.4f} -> tuned {tuned_test:.4f} ({diff_test:+.4f})
- **Conclusion:** {verdict}
- **Saved model:** `models/tuned_model.pkl`, retrained on all {len(X)} rows with the best parameters, so it also has no held-out score.
"""
card_path = MODELS_DIR / "model_card.md"
card = card_path.read_text(encoding="utf-8")
card = re.split(r"\n## Tuning \(W4D2\)", card)[0].rstrip() + "\n\n" + section
card_path.write_text(card, encoding="utf-8")
print("\nUpdated models/model_card.md and models/results.csv")
