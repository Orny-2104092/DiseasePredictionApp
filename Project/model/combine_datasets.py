import pandas as pd

# Load datasets
training = pd.read_csv("Project\data\Training.csv")
severity = pd.read_csv("Project\data\Symptom-severity.csv")

# Normalize column names
training.columns = [c.strip().lower().replace("-", "_") for c in training.columns]
severity.columns = [c.strip().lower().replace("-", "_") for c in severity.columns]

print("Training shape:", training.shape)
print("Severity shape:", severity.shape)

# Severity dictionary তৈরি
severity_dict = dict(zip(severity['symptom'], severity['weight']))

# Training dataset এ severity weight যোগ করা
def calculate_total_weight(row):
    total = 0
    for symptom, weight in severity_dict.items():
        if symptom in row.index and row[symptom] == 1:
            total += weight
    return total

training['total_weight'] = training.apply(calculate_total_weight, axis=1)

# Save combined dataset
training.to_csv("data\Combined_Training.csv", index=False)
print("✅ Combined dataset created and saved as Combined_Training.csv")
