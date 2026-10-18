"""W1D4: load the sample graph-definition dataset and inspect it."""
import pandas as pd

df = pd.read_csv("data/sample_dataset.csv")

print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\nData types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head(5).to_string(index=False))
