import os
from app import backend
from app import aitools  # Import your new tools
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, Union

app = FastAPI(title="invisibleai2026")

# --- 1. Define Request Models ---
# These tell FastAPI what JSON structure to expect for each endpoint

class TranscriptRequest(BaseModel):
    transcript: str
    info: Optional[Dict[str, Any]] = None

class QuizRequest(BaseModel):
    notes: Union[str, dict]  # Accepts either a string or your JSON dict
    num_questions: int
    info: Optional[Dict[str, Any]] = None

class MarkAnswerRequest(BaseModel):
    question: str
    user_answer: str
    marking_guide: str
    max_marks: int
    info: Optional[Dict[str, Any]] = None

# --- 2. Existing Routes ---

@app.get("/health")
def health():
    return {"status": "ok", "env": os.getenv("APP_ENV", "dev")}

@app.get("/")
def root():
    return backend.hello()

# --- 3. AI Tool Endpoints ---

@app.post("/api/summarise")
def summarise(request: TranscriptRequest):
    try:
        result = aitools.summarise_transcript(request.transcript, request.info)
        return {"status": "success", "data": result}
    except ValueError as e:
        # Returns a 400 status if the transcript is empty or invalid
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/quiz")
def quiz(request: QuizRequest):
    try:
        result = aitools.create_quiz(request.notes, request.num_questions, request.info)
        return {"status": "success", "data": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/mark")
def mark(request: MarkAnswerRequest):
    try:
        result = aitools.mark_answer(
            request.question,
            request.user_answer,
            request.marking_guide,
            request.max_marks,
            request.info
        )
        return {"status": "success", "data": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/flashcards")
def flashcards(request: TranscriptRequest):
    try:
        result = aitools.make_flashcards(request.transcript, request.info)
        return {"status": "success", "data": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")