from fastapi import FastAPI, Form, HTTPException, Header, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import re
import os
import pandas as pd 
import sqlalchemy
from pydantic import BaseModel, EmailStr, Field
from datetime import date, datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import select, func, text
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
# LOAD MODEL + SYMPTOMS - SECURE PATH HANDLING
# -----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.environ.get("MODEL_PATH")
if not MODEL_PATH:
    MODEL_PATH = os.path.join(BASE_DIR, "model", "disease_model.pkl")

SYMPTOM_LIST_PATH = os.environ.get("SYMPTOM_LIST_PATH")
if not SYMPTOM_LIST_PATH:
    SYMPTOM_LIST_PATH = os.path.join(BASE_DIR, "model", "symptom_list.pkl")

# Validate paths exist and are within allowed directories
def validate_path(path: str, allowed_dirs: list) -> bool:
    """Security check to prevent path traversal attacks"""
    real_path = os.path.realpath(path)
    for allowed in allowed_dirs:
        if real_path.startswith(os.path.realpath(allowed)):
            return True
    return False

ALLOWED_DATA_DIRS = [os.path.realpath(os.path.dirname(MODEL_PATH))]

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not validate_path(MODEL_PATH, ALLOWED_DATA_DIRS):
    raise PermissionError(f"Model path not in allowed directory: {MODEL_PATH}")

if not os.path.exists(SYMPTOM_LIST_PATH):
    raise FileNotFoundError(f"Symptom list not found: {SYMPTOM_LIST_PATH}")

if not validate_path(SYMPTOM_LIST_PATH, ALLOWED_DATA_DIRS):
    raise PermissionError(f"Symptom list path not in allowed directory: {SYMPTOM_LIST_PATH}")

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
DATA_DIR = os.environ.get("DATA_DIR")
if not DATA_DIR:
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
CSV_PATH = os.path.join(DATA_DIR, "symptom_precaution.csv")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

if not validate_path(CSV_PATH, [os.path.realpath(DATA_DIR)]):
    raise PermissionError(f"CSV path not in allowed directory: {CSV_PATH}")

print("CSV PATH:", CSV_PATH)  # debug

details_df = pd.read_csv(CSV_PATH)

print("Loaded columns:", details_df.columns.tolist())

# Load symptom description CSV for fallback
DESCRIPTION_CSV_PATH = os.path.join(DATA_DIR, "symptom_Description.csv")

if not os.path.exists(DESCRIPTION_CSV_PATH):
    raise FileNotFoundError(f"Description CSV not found: {DESCRIPTION_CSV_PATH}")

if not validate_path(DESCRIPTION_CSV_PATH, [os.path.realpath(DATA_DIR)]):
    raise PermissionError(f"Description CSV path not in allowed directory: {DESCRIPTION_CSV_PATH}")

print("DESCRIPTION CSV PATH:", DESCRIPTION_CSV_PATH)  # debug

description_df = pd.read_csv(DESCRIPTION_CSV_PATH)

print("Description CSV Loaded columns:", description_df.columns.tolist())



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
    fullName: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    dob: date
    gender: str = Field(..., min_length=1, max_length=20)
    nationality: str = Field(..., min_length=2, max_length=50)

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
    title: Optional[str] = Field(None, max_length=255)

class ChatOut(BaseModel):
    id: int
    user_id: int
    title: Optional[str]
    created_at: datetime

class CreateMessageIn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=10000)

class MessageOut(BaseModel):
    id: int
    chat_id: int
    user_id: Optional[int]
    role: str
    content: str
    created_at: datetime

class ChatWithMessagesOut(BaseModel):
    chat: ChatOut
    messages: List[MessageOut]

class UserProfileUpdate(BaseModel):
    fullName: Optional[str] = Field(None, min_length=2, max_length=100)
    dob: Optional[date] = None
    gender: Optional[str] = Field(None, min_length=1, max_length=20)
    nationality: Optional[str] = Field(None, min_length=2, max_length=50)

class PredictionIn(BaseModel):
    user_input: str = Field(..., min_length=1, max_length=2000)

class PredictionOut(BaseModel):
    user_input: str
    predicted_disease: str
    probability: Optional[float]
    matched_symptoms: List[str]

class DiseaseDetailsOut(BaseModel):
    disease: str
    description: str
    precautions: List[str]

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
async def predict_text(payload: PredictionIn, authorization: Optional[str] = Header(None)):
    user_id = None
    if authorization:
        try:
            user = await get_current_user(authorization)
            user_id = user["id"]
        except HTTPException:
            pass

    user_input = payload.user_input
    
    # Input validation and sanitization
    if not user_input or not user_input.strip():
        raise HTTPException(status_code=400, detail="User input cannot be empty")
    
    # Check for potentially malicious input patterns
    if len(user_input) > 2000:
        raise HTTPException(status_code=400, detail="Input too long (max 2000 characters)")
    
    # Check for SQL injection patterns
    dangerous_patterns = ["'", "\"", ";", "--", "/*", "*/", "DROP", "DELETE", "UPDATE", "INSERT", "UNION", "SELECT"]
    input_upper = user_input.upper()
    for pattern in dangerous_patterns:
        if pattern in input_upper:
            raise HTTPException(status_code=400, detail="Invalid characters in input")

    arr, matched = build_vector_from_text(user_input)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(arr)
        idx = np.argmax(probs)
        pred = model.classes_[idx]
        prob = float(probs[0][idx])
    else:
        pred = model.predict(arr)[0]
        prob = None

    return PredictionOut(
        user_input=user_input,
        predicted_disease=pred,
        probability=prob,
        matched_symptoms=matched
    )



@app.get("/get_details")
def get_details(disease: str = Query(..., min_length=1, max_length=200)):

    # Validate input
    if not disease or not disease.strip():
        raise HTTPException(status_code=400, detail="Disease parameter cannot be empty")
    
    # Sanitize input - only allow alphanumeric, spaces, and basic punctuation
    sanitized = re.sub(r"[^\w\s\-']", "", disease.strip())
    if len(sanitized) != len(disease.strip()):
        raise HTTPException(status_code=400, detail="Invalid characters in disease name")

    # Normalize all column names
    df = details_df.rename(columns=lambda x: x.strip().lower())

    # ensure disease column exists
    if "disease" not in df.columns:
        return {"error": "CSV missing 'Disease' column", "columns": list(df.columns)}

    # match disease
    row = df[df["disease"].str.strip().str.lower() == sanitized.lower()]

    # Get precautions from symptom_precaution.csv
    precautions = []
    if not row.empty:
        item = row.iloc[0]
        precautions = [
            str(item.get("precaution_1", "")),
            str(item.get("precaution_2", "")),
            str(item.get("precaution_3", "")),
            str(item.get("precaution_4", "")),
        ]
        precautions = [p for p in precautions if p and p.strip()]

    # Get description from symptom_Description.csv
    description = "No description found"
    desc_df = description_df.rename(columns=lambda x: x.strip().lower())
    
    if "disease" in desc_df.columns:
        desc_row = desc_df[desc_df["disease"].str.strip().str.lower() == sanitized.lower()]
        
        if not desc_row.empty:
            desc_item = desc_row.iloc[0]
            description = str(desc_item.get("description", "No description found"))

    return DiseaseDetailsOut(
        disease=disease,
        description=description,
        precautions=precautions
    )

# -----------------------------------------------------
# CHAT HISTORY ENDPOINTS
# -----------------------------------------------------

@app.get("/chats", response_model=List[ChatOut])
async def get_chats(authorization: Optional[str] = Header(None)):
    """Get all chats for the current user"""
    user = await get_current_user(authorization)
    user_id = user["id"]
    
    query = chats.select().where(chats.c.user_id == user_id).order_by(chats.c.created_at.desc())
    result = await database.fetch_all(query)
    return result

@app.post("/chats", response_model=ChatOut, status_code=201)
async def create_chat(payload: CreateChatIn, authorization: Optional[str] = Header(None)):
    """Create a new chat session"""
    user = await get_current_user(authorization)
    user_id = user["id"]
    
    # Generate title from first message or use default
    title = payload.title if payload.title else "New Chat"
    
    query = chats.insert().values(
        user_id=user_id,
        title=title
    ).returning(
        chats.c.id, chats.c.user_id, chats.c.title, chats.c.created_at
    )
    
    result = await database.fetch_one(query)
    return result

@app.get("/chats/{chat_id}", response_model=ChatWithMessagesOut)
async def get_chat_with_messages(chat_id: int, authorization: Optional[str] = Header(None)):
    """Get a specific chat with all its messages"""
    user = await get_current_user(authorization)
    user_id = user["id"]
    
    # Verify chat belongs to user
    chat_query = chats.select().where(chats.c.id == chat_id).where(chats.c.user_id == user_id)
    chat = await database.fetch_one(chat_query)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Get messages for this chat
    messages_query = messages.select().where(messages.c.chat_id == chat_id).order_by(messages.c.created_at.asc())
    chat_messages = await database.fetch_all(messages_query)
    
    return {"chat": dict(chat), "messages": [dict(m) for m in chat_messages]}

@app.delete("/chats/{chat_id}", status_code=204)
async def delete_chat(chat_id: int, authorization: Optional[str] = Header(None)):
    """Delete a chat and all its messages"""
    user = await get_current_user(authorization)
    user_id = user["id"]
    
    # Verify chat belongs to user before deleting
    chat_query = chats.select().where(chats.c.id == chat_id).where(chats.c.user_id == user_id)
    chat = await database.fetch_one(chat_query)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Delete the chat (messages will be cascade deleted due to FK)
    await database.execute(chats.delete().where(chats.c.id == chat_id))
    return None

@app.post("/chats/{chat_id}/messages", response_model=MessageOut, status_code=201)
async def create_message(chat_id: int, payload: CreateMessageIn, authorization: Optional[str] = Header(None)):
    """Add a message to a chat"""
    user = await get_current_user(authorization)
    user_id = user["id"]
    
    # Verify chat belongs to user
    chat_query = chats.select().where(chats.c.id == chat_id).where(chats.c.user_id == user_id)
    chat = await database.fetch_one(chat_query)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    query = messages.insert().values(
        chat_id=chat_id,
        user_id=user_id,
        role=payload.role,
        content=payload.content
    ).returning(
        messages.c.id, messages.c.chat_id, messages.c.user_id, 
        messages.c.role, messages.c.content, messages.c.created_at
    )
    
    result = await database.fetch_one(query)
    return result

# -----------------------------------------------------
# USER PROFILE ENDPOINTS
# -----------------------------------------------------

@app.get("/user/profile", response_model=UserOut)
async def get_user_profile(authorization: Optional[str] = Header(None)):
    """Get current user's profile"""
    user = await get_current_user(authorization)
    return {
        "id": user["id"],
        "full_name": user["full_name"],
        "email": user["email"],
        "dob": user["dob"],
        "gender": user["gender"],
        "nationality": user["nationality"],
        "created_at": user["created_at"]
    }

@app.put("/user/profile", response_model=UserOut)
async def update_user_profile(
    payload: UserProfileUpdate, 
    authorization: Optional[str] = Header(None)
):
    """Update current user's profile"""
    user = await get_current_user(authorization)
    user_id = user["id"]
    
    # Build update dictionary with only provided fields
    update_data = {}
    if payload.fullName is not None:
        update_data["full_name"] = payload.fullName
    if payload.dob is not None:
        update_data["dob"] = payload.dob
    if payload.gender is not None:
        update_data["gender"] = payload.gender
    if payload.nationality is not None:
        update_data["nationality"] = payload.nationality
    
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    query = users.update().where(users.c.id == user_id).values(**update_data).returning(
        users.c.id, users.c.full_name, users.c.email, users.c.dob,
        users.c.gender, users.c.nationality, users.c.created_at
    )
    
    result = await database.fetch_one(query)
    return result

@app.post("/user/change-password", status_code=200)
async def change_password(
    current_password: str = Form(...),
    new_password: str = Form(..., min_length=8),
    authorization: Optional[str] = Header(None)
):
    """Change user password"""
    user = await get_current_user(authorization)
    user_id = user["id"]
    
    # Get current user record with password hash
    query = users.select().where(users.c.id == user_id)
    user_record = await database.fetch_one(query)
    
    # Verify current password
    if not verify_password(current_password, user_record["password_hash"]):
        raise HTTPException(status_code=401, detail="Current password is incorrect")
    
    # Hash new password and update
    new_hash = hash_password(new_password)
    await database.execute(users.update().where(users.c.id == user_id).values(password_hash=new_hash))
    
    return {"message": "Password updated successfully"}

@app.get("/user/chat-stats", response_model=Dict[str, int])
async def get_user_chat_stats(authorization: Optional[str] = Header(None)):
    """Get statistics about user's chats"""
    user = await get_current_user(authorization)
    user_id = user["id"]
    
    # Count total chats
    chats_count_query = select(func.count()).select_from(chats).where(chats.c.user_id == user_id)
    chats_count = await database.fetch_val(chats_count_query)
    
    # Count total messages
    messages_count_query = select(func.count()).select_from(messages).where(messages.c.user_id == user_id)
    messages_count = await database.fetch_val(messages_count_query)
    
    return {
        "total_chats": chats_count,
        "total_messages": messages_count
    }

