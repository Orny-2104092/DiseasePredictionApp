# models.py
from sqlalchemy import Table, Column, Integer, String, Date, TIMESTAMP, func, Text, ForeignKey, Text
from .db import metadata

users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("full_name", Text, nullable=False),
    Column("dob", Date, nullable=False),
    Column("gender", Text, nullable=False),
    Column("nationality", Text, nullable=False),
    Column("email", Text, nullable=False, unique=True),
    Column("password_hash", Text, nullable=False),
    Column("created_at", TIMESTAMP(timezone=True), server_default=func.now()),
)
# models.py (append)
from sqlalchemy import ForeignKey

chats = Table(
    "chats",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("title", Text, nullable=True),
    Column("created_at", TIMESTAMP(timezone=True), server_default=func.now()),
)

messages = Table(
    "messages",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("chat_id", Integer, ForeignKey("chats.id", ondelete="CASCADE"), nullable=False),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
    Column("role", Text, nullable=False),
    Column("content", Text, nullable=False),
    Column("created_at", TIMESTAMP(timezone=True), server_default=func.now()),
)
