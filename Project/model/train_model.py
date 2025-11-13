import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load combined dataset
df = pd.read_csv("Project/data/Combined_Training.csv")

# X (features) & y (target)
X = df.drop(["prognosis"], axis=1)
y = df["prognosis"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model train
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Evaluate
acc = model.score(X_test, y_test)

print("✅ Model trained successfully!")
print(f"Accuracy: {acc * 100:.2f}%")

# Save model
with open("model/disease_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save symptom list for backend
with open("model/symptoms_list.txt", "w") as f:
    f.write("\n".join(list(X.columns)))

print("✅ Model & symptom list saved successfully.")


import pickle

# After training model
pickle.dump(list(X.columns), open("Project/model/symptom_list.pkl", "wb"))
print("Saved symptom_list.pkl with", len(X.columns), "symptoms")
