"""W2D2: find and handle missing values, then save data/cleaned_dataset.csv."""
import numpy as np
import pandas as pd

DROP_THRESHOLD = 40  # percent missing above which a column is dropped


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values: median for numeric, mode (or 'Unknown') for the rest."""
    df = df.copy()
    missing_pct = df.isnull().mean() * 100

    # Columns with too many missing values are dropped
    to_drop = missing_pct[missing_pct > DROP_THRESHOLD].index.tolist()
    if to_drop:
        print("Dropping columns (>40% missing):", to_drop)
        df = df.drop(columns=to_drop)

    for col in df.columns[df.isnull().any()]:
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            modes = df[col].mode()
            df[col] = df[col].fillna(modes[0] if len(modes) else "Unknown")
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/sample_dataset.csv")

    print("Missing values per column (before):")
    print(df.isnull().sum())

    cleaned = clean(df)

    assert cleaned.isnull().sum().sum() == 0, "Missing values remain after cleaning"
    print("\nCheck passed: zero missing values after cleaning")

    cleaned.to_csv("data/cleaned_dataset.csv", index=False)
    print("Saved data/cleaned_dataset.csv", cleaned.shape)

    # Self-test: blank out some values in a copy to prove clean() works
    test = df.copy()
    test.loc[1, "max_retries"] = np.nan
    test.loc[2, "condition"] = np.nan
    test["junk"] = [np.nan] * 5 + [1.0]  # 83% missing, should be dropped
    fixed = clean(test)
    assert fixed.isnull().sum().sum() == 0
    print("Self-test passed: clean() handles numeric, text, and high-missing columns")
