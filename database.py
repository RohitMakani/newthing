# backend/modules/database.py

from sqlalchemy import create_engine, Column, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import uuid
import os
import json

# ============================
# Database Setup
# ============================
MONGO_URL = os.getenv('MONGO_URL', 'sqlite:///./insightforge.db')
engine = create_engine(MONGO_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ============================
# Models
# ============================

class AnalysisSession(Base):
    __tablename__ = "analysis_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    task_type = Column(String, nullable=False)
    target_column = Column(String, nullable=False)
    dataset_info = Column(Text)  # JSON string
    results = Column(Text)       # JSON string


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    message = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    context_type = Column(String, default="general")

# ============================
# Table Initialization
# ============================
Base.metadata.create_all(bind=engine)

# ============================
# DB Session Dependency
# ============================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============================
# Database Manager Class
# ============================

class DatabaseManager:
    def __init__(self, engine):
        self.engine = engine

    def save_analysis_session(self, db: Session, session_data: dict) -> str:
        try:
            session = AnalysisSession(
                task_type=session_data.get('task_type'),
                target_column=session_data.get('target_column'),
                dataset_info=json.dumps(session_data.get('dataset_info', {})),
                results=json.dumps(session_data.get('results', {}))
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            return session.id
        except Exception as e:
            db.rollback()
            raise e

    def get_analysis_session(self, db: Session, session_id: str):
        session = db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
        if session:
            return {
                "id": session.id,
                "created_at": session.created_at,
                "task_type": session.task_type,
                "target_column": session.target_column,
                "dataset_info": json.loads(session.dataset_info or "{}"),
                "results": json.loads(session.results or "{}")
            }
        return None
