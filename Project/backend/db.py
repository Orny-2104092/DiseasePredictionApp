# db.py
import os
from databases import Database
from sqlalchemy import MetaData, create_engine

from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("NEON_DATABASE_URL not set in env")

# databases (async) object used by app code
database = Database(DATABASE_URL)

# for schema creation / alembic you can also use SQLAlchemy engine
engine = create_engine(DATABASE_URL)
metadata = MetaData()
