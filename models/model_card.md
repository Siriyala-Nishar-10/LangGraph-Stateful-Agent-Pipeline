# Model Card: LogisticRegression

Date: 2026-11-07
Project: LangGraph Stateful Agent Pipeline (Cynaris Solutions AI/ML internship)

## Model
- **Model:** LogisticRegression (scikit-learn), saved as `models/best_model.pkl` with joblib.
- **Selected by:** highest accuracy in `models/results.csv`. It tied with RandomForestClassifier on held-out accuracy; the simpler, more interpretable model was chosen.
- **Retraining:** after selection, the model was retrained on all 614 rows.

## Training data
- Kaggle "Loan Prediction Problem" dataset (`data/loan_approval.csv`), 614 rows, 13 columns.
- Target: `Loan_Status` (1 = approved: 422 rows, 0 = rejected: 192 rows).
- Features (17): scaled income and loan columns, engineered ratios (`TotalIncome`,
  `LoanAmount_to_Income`, `LogTotalIncome`), `Credit_History`, and one-hot encoded categories.
  Produced by `preprocessing/pipeline.py`.

## Metric
- **Held-out accuracy: 0.8618** on 123 test rows (20% stratified split, `random_state=42`),
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
