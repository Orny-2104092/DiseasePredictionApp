from fastapi import FastAPI, Form, HTTPException, Header, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import re
import os
import pandas as pd 
import sqlalchemy
from pydantic import BaseModel, EmailStr
from datetime import date, datetime
from typing import Optional, List, Dict
from sqlalchemy import select
from .auth_utils import decode_access_token
from .models import chats, messages, users  # ensure users imported


# existing imports: database, users, chats, messages, model, build_vector_from_text, details_df, etc.



# import database and models
from .db import database, metadata, engine
from .models import users
from .auth_utils import hash_password, verify_password, create_access_token


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
MODEL_PATH = r"D:/Files/MyAcademicFiles/3-2/DiseasePredictionApp-main/DiseasePredictionApp-main/Project/model/disease_model.pkl"
SYMPTOM_LIST_PATH = r"D:/Files/MyAcademicFiles/3-2/DiseasePredictionApp-main/DiseasePredictionApp-main/Project/model/symptom_list.pkl"


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
BASE_DIR = r"D:/Files/MyAcademicFiles/3-2/DiseasePredictionApp-main/DiseasePredictionApp-main/Project"
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

# create tables if not exists (optional)
def create_tables():
    metadata.create_all(bind=engine)

# Pydantic schemas
class RegisterIn(BaseModel):
    fullName: str
    email: EmailStr
    password: str
    dob: date
    gender: str
    nationality: str

class UserOut(BaseModel):
    id: int
    full_name: str
    email: EmailStr
    dob: date
    gender: str
    nationality: str
    created_at: datetime

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class LoginOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut

class CreateChatIn(BaseModel):
    title: Optional[str] = None

class ChatOut(BaseModel):
    id: int
    title: Optional[str]
    created_at: datetime

class CreateMessageIn(BaseModel):
    role: str  # "user" | "assistant"
    content: str

class MessageOut(BaseModel):
    id: int
    chat_id: int
    user_id: Optional[int]
    role: str
    content: str
    created_at: datetime

# -----------------------------------------------------
# ROUTES
# -----------------------------------------------------
@app.get("/")
def home():
    return {"message": "FastAPI running"}

# FastAPI startup/shutdown events to connect/disconnect database
@app.on_event("startup")
async def startup():
    # create tables if desired (useful for quick dev). Remove in prod if using migrations.
    create_tables()
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

# Register route
@app.post("/auth/register", status_code=201)
async def register(payload: RegisterIn):
    # check email exists
    query = users.select().where(users.c.email == payload.email)
    existing = await database.fetch_one(query)
    if existing:
        raise HTTPException(status_code=409, detail="Email is already registered")

    hashed = hash_password(payload.password)
    insert_query = users.insert().values(
        full_name=payload.fullName,
        dob=payload.dob,
        gender=payload.gender,
        nationality=payload.nationality,
        email=payload.email,
        password_hash=hashed,
    ).returning(
        users.c.id, users.c.full_name, users.c.email, users.c.dob,
        users.c.gender, users.c.nationality, users.c.created_at
    )

    created = await database.fetch_one(insert_query)

    return dict(created)

# Login route
@app.post("/auth/login", response_model=LoginOut)
async def login(payload: LoginIn):
    query = users.select().where(users.c.email == payload.email)
    user_row = await database.fetch_one(query)
    if not user_row:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(payload.password, user_row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(subject=str(user_row["id"]))
    user_out = {
        "id": user_row["id"],
        "full_name": user_row["full_name"],
        "email": user_row["email"],
        "dob": str(user_row["dob"]),
        "gender": user_row["gender"],
        "nationality": user_row["nationality"],
        "created_at": str(user_row["created_at"]),
    }

    return {"access_token": token, "user": user_out}

# dependency to extract current user id from Authorization header
async def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth scheme")
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = decode_access_token(token)
        sub = payload.get("sub")
        if not sub:
            raise HTTPException(status_code=401, detail="Invalid token")
        user_id = int(sub)
        q = users.select().where(users.c.id == user_id)
        row = await database.fetch_one(q)
        if not row:
            raise HTTPException(status_code=401, detail="User not found")
        return row  # RowMapping, use row['id']
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


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

