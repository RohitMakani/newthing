# backend/api.py

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

import uuid, os, json, cv2
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime
from pdf2image import convert_from_bytes

from .modules.enhanced_vision import CNNChartClassifier, EnhancedImageProcessor
from .modules.enhanced_eda import AutoEDAPipeline
from .modules.enhanced_ml import EnhancedMLPipeline
from .modules.enhanced_chat import IntelligentChatEngine
from .modules.database import get_db, AnalysisSession, ChatMessage
from .utils.pdf_utils import analyze_pdf_charts


UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
OUTPUT_FOLDER = os.getenv('OUTPUT_FOLDER', 'outputs')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_API_URL = os.getenv('GROQ_API_URL')
GROQ_MODEL = os.getenv('GROQ_MODEL')

router = APIRouter()

# Initialize engines
chart_classifier = CNNChartClassifier()
image_processor = EnhancedImageProcessor()
eda_pipeline = AutoEDAPipeline()
ml_pipeline = EnhancedMLPipeline()
chat_engine = IntelligentChatEngine(GROQ_API_KEY, GROQ_API_URL, GROQ_MODEL)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context_type: Optional[str] = "general"

class ChartAnalysisResponse(BaseModel):
    chart_type: str
    confidence: float
    extracted_data: Dict[str, Any]
    insights: List[str]


@router.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0"}


@router.post("/upload-dataset")
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
        df.columns = df.columns.str.strip()

        if target_column not in df.columns:
            raise HTTPException(400, f"Target column '{target_column}' not found")

        cleaned_df, eda_results = await eda_pipeline.run_analysis(df, task_type, target_column)
        model_results = await ml_pipeline.train_and_evaluate(cleaned_df, task_type, target_column)

        pdf_insights = None
        if pdf_file:
            pdf_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_{pdf_file.filename}")
            with open(pdf_path, "wb") as buffer:
                buffer.write(await pdf_file.read())
            pdf_insights = await analyze_pdf_charts(pdf_path, cleaned_df)

        session_data = AnalysisSession(
            id=session_id,
            task_type=task_type,
            target_column=target_column,
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


@router.post("/analyze-chart")
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


@router.post("/chat")
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


# Optional: you can move websocket, download, session routes here as well




