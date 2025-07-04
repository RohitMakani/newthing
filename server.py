import os
import sqlite3
import uuid
import json
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import cv2
import requests
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import pytesseract
from pdf2image import convert_from_bytes
from langdetect import detect
import pickle
from dotenv import load_dotenv

from backend.modules.enhanced_vision import CNNChartClassifier, EnhancedImageProcessor
from backend.modules.enhanced_eda import AutoEDAPipeline
from backend.modules.enhanced_ml import EnhancedMLPipeline
from backend.modules.enhanced_chat import IntelligentChatEngine
from backend.modules.database import DatabaseManager

load_dotenv()

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
OUTPUT_FOLDER = os.getenv('OUTPUT_FOLDER', 'outputs')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_API_URL = os.getenv('GROQ_API_URL')
GROQ_MODEL = os.getenv('GROQ_MODEL')
DATABASE_URL = os.getenv('MONGO_URL', 'sqlite:///./insightforge.db')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs('static/charts', exist_ok=True)

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI(title="InsightForge AI", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

chart_classifier = CNNChartClassifier()
image_processor = EnhancedImageProcessor()
eda_pipeline = AutoEDAPipeline()
ml_pipeline = EnhancedMLPipeline()
chat_engine = IntelligentChatEngine(GROQ_API_KEY, GROQ_API_URL, GROQ_MODEL)
db_manager = DatabaseManager(engine)

class AnalysisSession(Base):
    __tablename__ = "analysis_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    task_type = Column(String)
    target_column = Column(String)
    dataset_info = Column(Text)
    results = Column(Text)
    chat_history = Column(Text)

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String)
    message = Column(Text)
    response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    context_type = Column(String)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="InsightForge AI", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

chart_classifier = CNNChartClassifier()
image_processor = EnhancedImageProcessor()
eda_pipeline = AutoEDAPipeline()
ml_pipeline = EnhancedMLPipeline()
chat_engine = IntelligentChatEngine(GROQ_API_KEY, GROQ_API_URL, GROQ_MODEL)
db_manager = DatabaseManager(engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context_type: Optional[str] = "general"

class ChartAnalysisResponse(BaseModel):
    chart_type: str
    confidence: float
    extracted_data: Dict[str, Any]
    insights: List[str]

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.get("/")
async def root():
    return {"message": "Welcome to InsightForge.AI backend!", "status": "running"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "InsightForge AI", "version": "2.0.0"}

@app.post("/api/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    task_type: str = Form(...),
    target_column: str = Form(...),
    pdf_file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    try:
        session_id = str(uuid.uuid4())
        dataset_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_{file.filename}")
        with open(dataset_path, "wb") as buffer:
            buffer.write(await file.read())

        df = pd.read_csv(dataset_path)
        df.columns = df.columns.str.strip().str.replace(" ", "").str.title()
        normalized_target = target_column.strip().replace(" ", "").title()

        if normalized_target not in df.columns:
            raise HTTPException(400, f"Target column '{target_column}' not found")

        cleaned_df, eda_results = await eda_pipeline.run_analysis(df, task_type, normalized_target)
        model_results = await ml_pipeline.train_and_evaluate(cleaned_df, task_type, normalized_target)
        print("step 1")
        pdf_insights = None
        if pdf_file:
            pdf_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_{pdf_file.filename}")
            with open(pdf_path, "wb") as buffer:
                buffer.write(await pdf_file.read())
            pdf_insights = await analyze_pdf_charts(pdf_path, cleaned_df)

        session_data = AnalysisSession(
            id=session_id,
            task_type=task_type,
            target_column=normalized_target,
            dataset_info=json.dumps({
                "shape": cleaned_df.shape,
                "columns": list(cleaned_df.columns),
                "filename": file.filename
            }),
            results=json.dumps({
                "eda": eda_results,
                "ml": model_results,
                "pdf_insights": pdf_insights
            })
        )
        print("step 2")
        db.add(session_data)
        db.commit()

        return {
            "session_id": session_id,
            "eda_results": eda_results,
            "ml_results": model_results,
            "pdf_insights": pdf_insights
        }

    except Exception as e:
        print("error")
        raise HTTPException(500, f"Upload error: {str(e)}")

@app.post("/api/analyze-chart")
async def analyze_chart(file: UploadFile = File(...)):
    try:
        image_path = os.path.join(UPLOAD_FOLDER, f"chart_{uuid.uuid4().hex}_{file.filename}")
        with open(image_path, "wb") as buffer:
            buffer.write(await file.read())

        image = cv2.imread(image_path)
        chart_type, confidence = chart_classifier.classify_chart(image)
        extracted_data = image_processor.extract_chart_data(image, chart_type)
        insights = await chat_engine.generate_chart_insights(chart_type, extracted_data)

        return ChartAnalysisResponse(
            chart_type=chart_type,
            confidence=confidence,
            extracted_data=extracted_data,
            insights=insights
        )
    except Exception as e:
        raise HTTPException(500, f"Chart error: {str(e)}")

@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        context = {}
        if request.session_id:
            session = db.query(AnalysisSession).filter_by(id=request.session_id).first()
            if session:
                context = {
                    "task_type": session.task_type,
                    "target_column": session.target_column,
                    "results": json.loads(session.results)
                }

        response = await chat_engine.generate_response(request.message, context, request.context_type)

        chat_msg = ChatMessage(
            session_id=request.session_id,
            message=request.message,
            response=response,
            context_type=request.context_type
        )
        db.add(chat_msg)
        db.commit()

        return {"response": response}

    except Exception as e:
        raise HTTPException(500, f"Chat error: {str(e)}")

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
    if not session:
        raise HTTPException(404, "Session not found")

    chat_history = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).all()

    return {
        "session": {
            "id": session.id,
            "created_at": session.created_at,
            "task_type": session.task_type,
            "target_column": session.target_column,
            "dataset_info": json.loads(session.dataset_info) if session.dataset_info else {},
            "results": json.loads(session.results) if session.results else {}
        },
        "chat_history": [
            {
                "message": msg.message,
                "response": msg.response,
                "timestamp": msg.timestamp,
                "context_type": msg.context_type
            }
            for msg in chat_history
        ]
    }

@app.get("/api/sessions")
async def get_all_sessions(limit: int = 50, db: Session = Depends(get_db)):
    sessions = db.query(AnalysisSession).order_by(AnalysisSession.created_at.desc()).limit(limit).all()
    return {
        "sessions": [
            {
                "id": session.id,
                "created_at": session.created_at,
                "task_type": session.task_type,
                "target_column": session.target_column,
                "dataset_info": json.loads(session.dataset_info) if session.dataset_info else {}
            }
            for session in sessions
        ]
    }

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
    if not session:
        raise HTTPException(404, "Session not found")

    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
    db.delete(session)
    db.commit()

    return {"message": "Session deleted successfully"}

@app.get("/api/download/{session_id}/{file_type}")
async def download_file(session_id: str, file_type: str):
    try:
        file_mapping = {
            "cleaned_data": f"{OUTPUT_FOLDER}/cleaned_data_{session_id}.csv",
            "eda_report": f"{OUTPUT_FOLDER}/eda_report_{session_id}.pdf",
            "model": f"{OUTPUT_FOLDER}/model_{session_id}.pkl",
            "chat_history": f"{OUTPUT_FOLDER}/chat_history_{session_id}.txt"
        }

        if file_type not in file_mapping:
            raise HTTPException(400, "Invalid file type")

        file_path = file_mapping[file_type]
        if not os.path.exists(file_path):
            raise HTTPException(404, "File not found")

        return FileResponse(file_path, filename=f"{file_type}_{session_id}")

    except Exception as e:
        raise HTTPException(500, f"Download failed: {str(e)}")

@app.websocket("/api/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)

            response = await chat_engine.generate_response(
                message_data["message"],
                {},
                message_data.get("context_type", "general")
            )

            await manager.send_personal_message(
                json.dumps({"response": response, "timestamp": str(datetime.now())}),
                websocket
            )

    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def analyze_pdf_charts(pdf_path: str, dataset: pd.DataFrame) -> Dict[str, Any]:
    try:
        with open(pdf_path, 'rb') as f:
            images = convert_from_bytes(f.read())

        chart_data = []
        for i, page in enumerate(images):
            np_img = np.array(page)
            img_cv = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            chart_regions = image_processor.extract_chart_regions(img_cv)

            for j, chart_img in enumerate(chart_regions):
                chart_type, confidence = chart_classifier.classify_chart(chart_img)
                extracted_data = image_processor.extract_chart_data(chart_img, chart_type)
                insights = await chat_engine.generate_chart_insights(chart_type, extracted_data, dataset)

                chart_data.append({
                    "page": i + 1,
                    "chart": j + 1,
                    "type": chart_type,
                    "confidence": confidence,
                    "data": extracted_data,
                    "insights": insights
                })

        return {
            "total_charts": len(chart_data),
            "charts": chart_data,
            "summary_insights": await chat_engine.generate_summary_insights(chart_data, dataset)
        }

    except Exception as e:
        return {"error": f"PDF analysis failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
