from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import re
import os
import pandas as pd

# -----------------------------------------------------
# INIT APP FIRST (very important!)
# -----------------------------------------------------
app = FastAPI(title="Disease Prediction API", version="1.0")

# -----------------------------------------------------
# ENABLE CORS BEFORE ANY ROUTES
# -----------------------------------------------------
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
# LOAD MODEL + SYMPTOMS
# -----------------------------------------------------
MODEL_PATH = r"D:\DiseasePredictionApp\Project\model\disease_model.pkl"
SYMPTOM_LIST_PATH = r"D:\DiseasePredictionApp\Project\model\symptom_list.pkl"


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model missing: " + MODEL_PATH)

if not os.path.exists(SYMPTOM_LIST_PATH):
    raise FileNotFoundError("Symptom list missing: " + SYMPTOM_LIST_PATH)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SYMPTOM_LIST_PATH, "rb") as f:
    SYMPTOMS = pickle.load(f)

SYMPTOMS = [s.lower().strip() for s in SYMPTOMS]

# Load features from model
model_feature_names = None
if hasattr(model, "feature_names_in_"):
    model_feature_names = [s.lower().strip() for s in model.feature_names_in_]

# -----------------------------------------------------
# CSV for description + precautions
# -----------------------------------------------------
BASE_DIR = r"D:\DiseasePredictionApp\Project"
CSV_PATH = os.path.join(BASE_DIR, "data", "symptom_precaution.csv")

print("CSV PATH:", CSV_PATH)  # debug

details_df = pd.read_csv(CSV_PATH)

print("Loaded columns:", details_df.columns.tolist())



# -----------------------------------------------------
# HELPERS
# -----------------------------------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_vector_from_text(text):
    text = preprocess(text)
    found = []

    for s in SYMPTOMS:
        normalized = s.replace("_", " ")
        if normalized in text:
            found.append(s)

    vec = [1 if s in found else 0 for s in SYMPTOMS]
    arr = np.array(vec).reshape(1, -1)

    if model_feature_names:
        mapping = {s: (1 if s in found else 0) for s in SYMPTOMS}
        aligned = []
        for name in model_feature_names:
            key = name.replace(" ", "_")
            aligned.append(mapping.get(key, 0))
        arr = np.array(aligned).reshape(1, -1)

    return arr, found


# -----------------------------------------------------
# ROUTES
# -----------------------------------------------------
@app.get("/")
def home():
    return {"message": "FastAPI running"}

@app.post("/predict_text")
def predict_text(user_input: str = Form(...)):
    arr, matched = build_vector_from_text(user_input)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(arr)
        idx = np.argmax(probs)
        pred = model.classes_[idx]
        prob = float(probs[0][idx])
    else:
        pred = model.predict(arr)[0]
        prob = None

    return {
        "user_input": user_input,
        "predicted_disease": pred,
        "probability": prob,
        "matched_symptoms": matched,
    }


@app.get("/get_details")
def get_details(disease: str):

    # Normalize all column names
    df = details_df.rename(columns=lambda x: x.strip().lower())

    # ensure disease column exists
    if "disease" not in df.columns:
        return {"error": "CSV missing 'Disease' column", "columns": list(df.columns)}

    # match disease
    row = df[df["disease"].str.strip().str.lower() == disease.lower()]

    if row.empty:
        return {
            "disease": disease,
            "description": "No description found",
            "precautions": []
        }

    item = row.iloc[0]

    # extract precautions safely
    precautions = [
        item.get("precaution_1", ""),
        item.get("precaution_2", ""),
        item.get("precaution_3", ""),
        item.get("precaution_4", ""),
    ]

    return {
        "disease": disease,
        "description": item.get("description", "No description found"),
        "precautions": precautions
    }
