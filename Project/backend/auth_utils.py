# auth_utils.py (robust: avoids bcrypt 72-byte errors)
import os
import hashlib
from datetime import datetime, timedelta
from passlib.context import CryptContext
import jwt
from dotenv import load_dotenv

load_dotenv()

# prefer bcrypt_sha256 but we add a defensive pre-hash for very long passwords
pwd_context = CryptContext(
    schemes=["bcrypt_sha256", "bcrypt"],
    deprecated="auto",
)

JWT_SECRET = os.environ.get("JWT_SECRET", "please_change_me")
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_EXPIRES_SECONDS = int(os.environ.get("JWT_EXPIRES_SECONDS", 3600))

# max safe raw length for bcrypt is 72 bytes; choose threshold well under that
_BCRYPT_MAX_BYTES = 72

def _should_prehash(plain: str) -> bool:
    if not isinstance(plain, str):
        return True
    try:
        return len(plain.encode("utf-8")) > _BCRYPT_MAX_BYTES
    except Exception:
        return True

def _prehash_sha256(plain: str) -> str:
    """Return hex digest (64 chars) of sha256 of plain password."""
    return hashlib.sha256(plain.encode("utf-8")).hexdigest()

def hash_password(password: str) -> str:
    if not isinstance(password, str):
        raise TypeError("password must be a string")

    # If password is too long for some bcrypt implementations, prehash first.
    if _should_prehash(password):
        # prehash to hex (64 bytes) then hash that value (safe)
        return pwd_context.hash(_prehash_sha256(password))
    # otherwise hash the raw password (passlib will choose bcrypt_sha256 if configured)
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    # Try normal verification first
    try:
        if pwd_context.verify(plain_password, hashed_password):
            return True
    except ValueError:
        # some backends may raise on long inputs — fall through to prehash branch
        pass
    # If the password is long or verify failed, also try verifying the prehashed form
    if _should_prehash(plain_password):
        try:
            return pwd_context.verify(_prehash_sha256(plain_password), hashed_password)
        except Exception:
            return False
    return False

def create_access_token(subject: str, expires_seconds: int = None) -> str:
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
