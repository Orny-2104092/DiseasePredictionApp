import os
import hashlib
from datetime import datetime, timedelta
import jwt
from dotenv import load_dotenv

load_dotenv()

JWT_SECRET = os.environ.get("JWT_SECRET", "please_change_me")
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_EXPIRES_SECONDS = int(os.getenv("JWT_EXPIRES_SECONDS", 3600))

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hash_password(plain_password) == hashed_password

def create_access_token(subject: str, expires_seconds: int | None = None) -> str:
    if expires_seconds is None:
        expires_seconds = JWT_EXPIRES_SECONDS
    expire = datetime.utcnow() + timedelta(seconds=expires_seconds)
    payload = {"sub": subject, "exp": expire}
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    if isinstance(token, bytes):
        return token.decode("utf-8")
    return token

def decode_access_token(token: str):
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
