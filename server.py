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
from fastapi.responses import FileResponse, HTMLResponse
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

# Import local modules
from enhanced_vision import CNNChartClassifier, EnhancedImageProcessor
from enhanced_eda import AutoEDAPipeline
from enhanced_ml import EnhancedMLPipeline
from enhanced_chat import IntelligentChatEngine
from database import DatabaseManager, get_db, AnalysisSession, ChatMessage

load_dotenv()

# Environment variables
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
OUTPUT_FOLDER = os.getenv('OUTPUT_FOLDER', 'outputs')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_API_URL = os.getenv('GROQ_API_URL')
GROQ_MODEL = os.getenv('GROQ_MODEL')
DATABASE_URL = os.getenv('MONGO_URL', 'sqlite:///./insightforge.db')

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs('static/charts', exist_ok=True)

# Database setup
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Initialize FastAPI app
app = FastAPI(title="InsightForge AI", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize components
chart_classifier = CNNChartClassifier()
image_processor = EnhancedImageProcessor()
eda_pipeline = AutoEDAPipeline()
ml_pipeline = EnhancedMLPipeline()
chat_engine = IntelligentChatEngine(GROQ_API_KEY, GROQ_API_URL, GROQ_MODEL)
db_manager = DatabaseManager(engine)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context_type: Optional[str] = "general"

class ChartAnalysisResponse(BaseModel):
    chart_type: str
    confidence: float
    extracted_data: Dict[str, Any]
    insights: List[str]

# WebSocket connection manager
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

# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>InsightForge AI</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
            .section h3 { color: #555; margin-top: 0; }
            input[type="file"], input[type="text"], select { width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ddd; border-radius: 4px; }
            button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px 0; }
            button:hover { background-color: #0056b3; }
            .result { margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 4px; }
            .error { background-color: #f8d7da; border: 1px solid #f5c6cb; }
            .success { background-color: #d4edda; border: 1px solid #c3e6cb; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>InsightForge AI - Data Analysis Platform</h1>
            
            <div class="section">
                <h3>📊 Upload Dataset for Analysis</h3>
                <form id="datasetForm" enctype="multipart/form-data">
                    <input type="file" id="dataset" name="file" accept=".csv" required>
                    <select id="taskType" name="task_type" required>
                        <option value="">Select Task Type</option>
                        <option value="classification">Classification</option>
                        <option value="regression">Regression</option>
                    </select>
                    <input type="text" id="targetColumn" name="target_column" placeholder="Target Column Name" required>
                    <input type="file" id="pdfFile" name="pdf_file" accept=".pdf" placeholder="Optional: PDF with charts">
                    <button type="submit">🚀 Start Analysis</button>
                </form>
                <div id="datasetResult" class="result" style="display: none;"></div>
            </div>

            <div class="section">
                <h3>🎯 Analyze Chart Image</h3>
                <form id="chartForm" enctype="multipart/form-data">
                    <input type="file" id="chartImage" name="file" accept="image/*" required>
                    <button type="submit">📈 Analyze Chart</button>
                </form>
                <div id="chartResult" class="result" style="display: none;"></div>
            </div>

            <div class="section">
                <h3>💬 Chat with AI Assistant</h3>
                <input type="text" id="chatMessage" placeholder="Ask me about your data analysis...">
                <input type="text" id="sessionId" placeholder="Session ID (optional)">
                <button onclick="sendChat()">💬 Send Message</button>
                <div id="chatResult" class="result" style="display: none;"></div>
            </div>

            <div class="section">
                <h3>📋 Recent Analysis Sessions</h3>
                <button onclick="loadSessions()">📊 Load Sessions</button>
                <div id="sessionsResult" class="result" style="display: none;"></div>
            </div>
        </div>

        <script>
            // Dataset Upload
            document.getElementById('datasetForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const formData = new FormData();
                formData.append('file', document.getElementById('dataset').files[0]);
                formData.append('task_type', document.getElementById('taskType').value);
                formData.append('target_column', document.getElementById('targetColumn').value);
                
                const pdfFile = document.getElementById('pdfFile').files[0];
                if (pdfFile) {
                    formData.append('pdf_file', pdfFile);
                }

                try {
                    const response = await fetch('/api/upload-dataset', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    const resultDiv = document.getElementById('datasetResult');
                    
                    if (response.ok) {
                        resultDiv.className = 'result success';
                        resultDiv.innerHTML = `
                            <h4>✅ Analysis Complete!</h4>
                            <p><strong>Session ID:</strong> ${result.session_id}</p>
                            <p><strong>EDA Results:</strong> ${JSON.stringify(result.eda_results.statistics || {}, null, 2)}</p>
                            <p><strong>ML Results:</strong> ${JSON.stringify(result.ml_results.best_model || {}, null, 2)}</p>
                        `;
                    } else {
                        resultDiv.className = 'result error';
                        resultDiv.innerHTML = `<p>❌ Error: ${result.detail}</p>`;
                    }
                    resultDiv.style.display = 'block';
                } catch (error) {
                    const resultDiv = document.getElementById('datasetResult');
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `<p>❌ Error: ${error.message}</p>`;
                    resultDiv.style.display = 'block';
                }
            });

            // Chart Analysis
            document.getElementById('chartForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const formData = new FormData();
                formData.append('file', document.getElementById('chartImage').files[0]);

                try {
                    const response = await fetch('/api/analyze-chart', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    const resultDiv = document.getElementById('chartResult');
                    
                    if (response.ok) {
                        resultDiv.className = 'result success';
                        resultDiv.innerHTML = `
                            <h4>📊 Chart Analysis Results</h4>
                            <p><strong>Chart Type:</strong> ${result.chart_type}</p>
                            <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                            <p><strong>Insights:</strong></p>
                            <ul>${result.insights.map(insight => `<li>${insight}</li>`).join('')}</ul>
                        `;
                    } else {
                        resultDiv.className = 'result error';
                        resultDiv.innerHTML = `<p>❌ Error: ${result.detail}</p>`;
                    }
                    resultDiv.style.display = 'block';
                } catch (error) {
                    const resultDiv = document.getElementById('chartResult');
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `<p>❌ Error: ${error.message}</p>`;
                    resultDiv.style.display = 'block';
                }
            });

            // Chat Function
            async function sendChat() {
                const message = document.getElementById('chatMessage').value;
                const sessionId = document.getElementById('sessionId').value;
                
                if (!message.trim()) return;

                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: message,
                            session_id: sessionId || null,
                            context_type: 'general'
                        })
                    });
                    
                    const result = await response.json();
                    const resultDiv = document.getElementById('chatResult');
                    
                    if (response.ok) {
                        resultDiv.className = 'result success';
                        resultDiv.innerHTML = `
                            <h4>🤖 AI Response</h4>
                            <p><strong>You:</strong> ${message}</p>
                            <p><strong>AI:</strong> ${result.response}</p>
                        `;
                    } else {
                        resultDiv.className = 'result error';
                        resultDiv.innerHTML = `<p>❌ Error: ${result.detail}</p>`;
                    }
                    resultDiv.style.display = 'block';
                    document.getElementById('chatMessage').value = '';
                } catch (error) {
                    const resultDiv = document.getElementById('chatResult');
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `<p>❌ Error: ${error.message}</p>`;
                    resultDiv.style.display = 'block';
                }
            }

            // Load Sessions
            async function loadSessions() {
                try {
                    const response = await fetch('/api/sessions');
                    const result = await response.json();
                    const resultDiv = document.getElementById('sessionsResult');
                    
                    if (response.ok) {
                        resultDiv.className = 'result success';
                        resultDiv.innerHTML = `
                            <h4>📋 Recent Sessions</h4>
                            ${result.sessions.map(session => `
                                <div style="border: 1px solid #ddd; padding: 10px; margin: 5px 0; border-radius: 4px;">
                                    <p><strong>ID:</strong> ${session.id}</p>
                                    <p><strong>Task:</strong> ${session.task_type}</p>
                                    <p><strong>Target:</strong> ${session.target_column}</p>
                                    <p><strong>Created:</strong> ${new Date(session.created_at).toLocaleString()}</p>
                                </div>
                            `).join('')}
                        `;
                    } else {
                        resultDiv.className = 'result error';
                        resultDiv.innerHTML = `<p>❌ Error: ${result.detail}</p>`;
                    }
                    resultDiv.style.display = 'block';
                } catch (error) {
                    const resultDiv = document.getElementById('sessionsResult');
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `<p>❌ Error: ${error.message}</p>`;
                    resultDiv.style.display = 'block';
                }
            }

            // Enter key for chat
            document.getElementById('chatMessage').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendChat();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

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
            raise HTTPException(400, f"Target column '{target_column}' not found. Available columns: {list(df.columns)}")

        cleaned_df, eda_results = await eda_pipeline.run_analysis(df, task_type, normalized_target)
        model_results = await ml_pipeline.train_and_evaluate(cleaned_df, task_type, normalized_target)

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
        db.add(session_data)
        db.commit()

        return {
            "session_id": session_id,
            "eda_results": eda_results,
            "ml_results": model_results,
            "pdf_insights": pdf_insights
        }

    except Exception as e:
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
            session = db.query(AnalysisSession).filter(AnalysisSession.id == request.session_id).first()
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
