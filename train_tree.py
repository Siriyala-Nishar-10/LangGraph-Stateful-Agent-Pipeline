"""W3D3: train a shallow decision tree and export human-readable rules."""
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save the figure to a file without opening a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from preprocessing.pipeline import RAW_CSV, preprocess

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_NAME = "DecisionTreeClassifier"

# 1. Same data and same 80/20 stratified split as the earlier models
X, y = preprocess(pd.read_csv(RAW_CSV))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Shallow tree (max_depth=5 keeps the rules readable)
model = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train, y_train)
accuracy = round(accuracy_score(y_test, model.predict(X_test)), 4)
print(f"{MODEL_NAME} accuracy (max_depth=5): {accuracy:.3f}")

# 3. Export the rules as text (class 1 = approved, 0 = rejected)
rules = export_text(model, feature_names=list(X.columns))
MODELS_DIR.mkdir(exist_ok=True)
(MODELS_DIR / "decision_tree_rules.txt").write_text(rules, encoding="utf-8")
print("\nFirst rules:")
print("\n".join(rules.splitlines()[:12]))

# 4. Plot and save feature importances (features the tree actually uses)
importances = pd.Series(model.feature_importances_, index=X.columns)
used = importances[importances > 0].sort_values()
print("\nFeatures used by the tree:")
print(used.sort_values(ascending=False).round(3).to_string())

fig, ax = plt.subplots(figsize=(8, 5))
used.plot(kind="barh", ax=ax)
ax.set_title("Decision tree feature importances (max_depth=5)")
ax.set_xlabel("importance")
fig.tight_layout()
fig.savefig(MODELS_DIR / "feature_importances.png", dpi=150)

# 5. Append the score to models/results.csv (replace an earlier run of the same model)
results_path = MODELS_DIR / "results.csv"
results = pd.read_csv(results_path)
row = pd.DataFrame([{
    "model_name": MODEL_NAME,
    "metric_name": "accuracy",
    "score": accuracy,
    "date": date.today().isoformat(),
}])
results = results[~((results["model_name"] == MODEL_NAME) & (results["metric_name"] == "accuracy"))]
pd.concat([results, row], ignore_index=True).to_csv(results_path, index=False)
print("\nSaved models/decision_tree_rules.txt, models/feature_importances.png and updated models/results.csv")
print(pd.read_csv(results_path).to_string(index=False))
