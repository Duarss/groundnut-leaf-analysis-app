# app/database/db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import Config

# SQLAlchemy engine
engine = create_engine(
    Config.database_url(),
    pool_pre_ping=True,
    future=True,
)

# Session factory
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
)

# Base untuk model ORM
Base = declarative_base()


def get_db():
    """
    Dependency / helper untuk ambil session DB
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
