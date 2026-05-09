import pandas as pd

# Load dataset from project root
df = pd.read_csv("data/Malaria-Data.csv")

print("=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

print("\nRaw counts:")
print(df["severe_maleria"].value_counts())

print("\nPercent distribution:")
print(df["severe_maleria"].value_counts(normalize=True))

print("\n" + "=" * 60)
print("CORRELATION WITH TARGET")
print("=" * 60)

corr = df.corr(numeric_only=True)["severe_maleria"]
print(corr.sort_values(ascending=False))