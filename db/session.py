"""
db/session.py
Database engine and session factory. Reads DATABASE_URL from env or defaults to sqlite local file.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# Import Base so the module importing session has access to ORM metadata
from db.models import Base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./triple_riding.db")  # default local sqlite

# Use connect_args for sqlite to avoid threading issues in dev; for Postgres remove connect_args.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

_engine = create_engine(DATABASE_URL, echo=False, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def get_engine():
    return _engine


def create_all_tables():
    try:
        Base.metadata.create_all(bind=_engine)
    except SQLAlchemyError as e:
        raise RuntimeError(f"Failed to create tables: {e}")
