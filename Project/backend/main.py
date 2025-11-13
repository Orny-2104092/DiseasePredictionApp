from fastapi import FastAPI, Form, HTTPException
import pickle
import numpy as np
import re
import os

app = FastAPI(title="Disease Prediction API", version="1.0")

# ---------------------------
# CONFIG — path adjust করা যাবে যদি দরকার হয়
# ---------------------------
MODEL_PATH = "Project/model/disease_model.pkl"
SYMPTOM_LIST_PATH = "Project/model/symptom_list.pkl"  # তুমি আগে যেভাবে save করেছো
# ---------------------------

# চেক করে নেই ফাইল আছে কি না
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"MODEL NOT FOUND at: {MODEL_PATH}")

if not os.path.exists(SYMPTOM_LIST_PATH):
    raise FileNotFoundError(f"Symptom list not found: {SYMPTOM_LIST_PATH}")

# মডেল লোড
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# symptom list লোড (এই ফাইলে তোমার ডোমেনের সব symptom নাম থাকবে, এক per line)
with open(SYMPTOM_LIST_PATH, "rb") as f:
    try:
        SYMPTOMS = pickle.load(f)
    except Exception:
        # যদি pickle না হয়, মুছে ফেলো — পরে পার্স করার জন্য
        raise RuntimeError("Could not load symptom list (bad pickle).")

# ensure it's a list
if not isinstance(SYMPTOMS, (list, tuple)):
    raise RuntimeError("SYMPTOMS must be a list of symptom names.")

# Clean symptoms: lower + strip
SYMPTOMS = [str(s).strip().lower() for s in SYMPTOMS]
TOTAL_SYMPTOMS = len(SYMPTOMS)
print(f"Total Symptoms Loaded: {TOTAL_SYMPTOMS}")

# যদি মডেল এ feature names থাকে, সেই অনুযায়ী validate করব
model_feature_names = None
if hasattr(model, "feature_names_in_"):
    model_feature_names = list(model.feature_names_in_)
    # normalize
    model_feature_names = [str(x).strip().lower() for x in model_feature_names]
    print(f"Loaded feature names from model: {len(model_feature_names)} features.")

# preprocessing helper
def preprocess(text: str) -> str:
    text = text.lower()
    # punctuation বাদ
    text = re.sub(r"[^\w\s]", " ", text)
    # multiple spaces -> single
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_vector_from_text(text: str):
    """
    Text থেকে SYMPTOMS লিস্ট ব্যবহার করে একটি binary vector বানাবে,
    vector length হবে len(SYMPTOMS) (বা model_feature_names যদি সেটা ভিন্ন হয়)।
    """
    text = preprocess(text)
    found = []
    # For phrase symptoms, check exact word/phrase presence
    for s in SYMPTOMS:
        # compare normalized forms; symptoms may have underscores in saved list
        normalized_sym = s.replace("_", " ").strip()
        if normalized_sym and ((" " + normalized_sym + " ") in (" " + text + " ") or normalized_sym in text.split()):
            found.append(s)
    # Build vector aligned to SYMPTOMS order
    vec = [1 if s in found else 0 for s in SYMPTOMS]
    arr = np.array(vec).reshape(1, -1)

    # If model expects a different order/columns (feature_names_in_), reorder/create accordingly
    if model_feature_names is not None:
        # build a vector aligned to model_feature_names
        # create a dict symptom->value using SYMPTOMS order
        symptom_to_val = {s: (1 if s in found else 0) for s in SYMPTOMS}
        aligned = []
        for fname in model_feature_names:
            # try direct match, then underscore/space variants
            key = fname.strip().lower()
            if key in symptom_to_val:
                aligned.append(symptom_to_val[key])
            else:
                # fallback: if fname has spaces but SYMPTOMS has underscores
                alt = key.replace(" ", "_")
                if alt in symptom_to_val:
                    aligned.append(symptom_to_val[alt])
                else:
                    # If feature not in symptom list, assume 0
                    aligned.append(0)
        arr = np.array(aligned).reshape(1, -1)
    return arr, found

@app.get("/")
def home():
    return {"message": "Hello! FastAPI is running successfully ✅"}

@app.post("/predict_text")
def predict_text(user_input: str = Form(...)):
    """
    Receives form-data field 'user_input' (string).
    Returns predicted disease, probability, and matched symptoms.
    """
    if not isinstance(user_input, str) or user_input.strip() == "":
        raise HTTPException(status_code=422, detail="user_input must be a non-empty string.")

    arr, matched = build_vector_from_text(user_input)

    # safety check: feature length must match model expectation
    expected = None
    try:
        # sklearn stores n_features_in_ after fit
        expected = getattr(model, "n_features_in_", None)
        if expected is None and hasattr(model, "feature_names_in_"):
            expected = len(model.feature_names_in_)
    except Exception:
        expected = None

    if expected is not None and arr.shape[1] != expected:
        raise HTTPException(
            status_code=500,
            detail=f"Feature-length mismatch: vector has {arr.shape[1]} features, model expects {expected}."
        )

    # predict
    try:
        # if predict_proba available, use it for probability
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(arr)
            # argmax index gives predicted class index
            idx = np.argmax(probs, axis=1)[0]
            pred_label = model.classes_[idx] if hasattr(model, "classes_") else str(idx)
            pred_prob = float(probs[0][idx])
        else:
            pred = model.predict(arr)
            pred_label = str(pred[0])
            pred_prob = None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

    return {
        "user_input": user_input,
        "predicted_disease": str(pred_label),
        "probability": pred_prob,
        "matched_symptoms": matched
    }
