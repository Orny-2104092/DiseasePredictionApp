import pandas as pd

# Load dataset
df = pd.read_csv("Project/data/Training.csv")

print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist()[:10])
print("\nSample Data:\n", df.head())
print("\nMissing Values:", df.isnull().sum().sum())
