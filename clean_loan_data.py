"""W2D2 (loan dataset): clean data/loan_approval.csv and save data/cleaned_loan_dataset.csv."""
import pandas as pd

from clean_data import clean

if __name__ == "__main__":
    df = pd.read_csv("data/loan_approval.csv")

    print("Shape:", df.shape)
    print("\nMissing values per column (before):")
    before = df.isnull().sum()
    print(before[before > 0])
    print("Columns above 5% missing:", df.columns[df.isnull().mean() * 100 > 5].tolist())

    cleaned = clean(df)

    assert cleaned.isnull().sum().sum() == 0, "Missing values remain after cleaning"
    print("\nCheck passed: zero missing values after cleaning")

    cleaned.to_csv("data/cleaned_loan_dataset.csv", index=False)
    print("Saved data/cleaned_loan_dataset.csv", cleaned.shape)
