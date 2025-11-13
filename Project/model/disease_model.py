import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("Project/data/Training.csv")

# Separate features and labels
X = df.drop(columns=["prognosis"])
y = df["prognosis"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("✅ Model trained successfully!")
print("Accuracy:", acc)

# Save model
with open("disease_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("💾 Model saved as disease_model.pkl")
