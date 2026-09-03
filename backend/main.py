import os
import shutil
import uuid
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Internal Service Imports
from services.ingest import supabase
from services.tasks import process_document_task
from services.rag import (
    answer_question,
    stream_answer_question,
    generate_document_summary,
    DocumentSummary
)

# 1. App Initialization & Metadata
app = FastAPI(
    title="DocuMind AI v2",
    description="Enterprise RAG engine powered by Hybrid Search, Gemini 2.5 Flash, and Celery.",
    version="2.0.0"
)

# 2. CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# 3. Pydantic Schemas (Data Contracts)

class ChatRequest(BaseModel):
    document_id: str = Field(..., description="UUID of the indexed document")
    question: str = Field(..., min_length=1, description="User question to query against the document")

class Citation(BaseModel):
    source_id: int
    chunk_id: str
    similarity: float
    rrf_score: Optional[float] = 0.0
    preview: str

class ChatResponse(BaseModel):
    answer: str
    confidence_score: float
    citations: List[Citation]

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str

class StatusResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    created_at: Optional[str] = None


# 4. API Endpoints

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "HEALTHY", "engine": "DocuMind AI v2"}


@app.post(
    "/api/v1/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Documents"]
)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Only PDF documents are accepted."
        )

    doc_id = str(uuid.uuid4())
    storage_path = f"{doc_id}_{file.filename}"

    try:
        # Read file bytes directly into memory
        file_bytes = await file.read()

        # Upload directly to Supabase Storage bucket 'documents'
        supabase.storage.from_("documents").upload(
            path=storage_path,
            file=file_bytes,
            file_options={"content-type": "application/pdf"}
        )

        # Record in Supabase DB
        supabase.table("documents").insert({
            "id": doc_id,
            "filename": file.filename,
            "status": "PROCESSING"
        }).execute()

        # Enqueue task passing the Supabase Storage path/key
        process_document_task.delay(doc_id=doc_id, file_path=storage_path)

        return UploadResponse(
            document_id=doc_id,
            filename=file.filename,
            status="PROCESSING",
            message="Document uploaded and queued for background ingestion."
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@app.get(
    "/api/v1/documents/{doc_id}/status",
    response_model=StatusResponse,
    tags=["Documents"]
)
async def get_document_status(doc_id: str):
    response = supabase.table("documents").select("*").eq("id", doc_id).execute()
    if not response.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' does not exist."
        )

    doc = response.data[0]
    return StatusResponse(
        document_id=doc["id"],
        filename=doc["filename"],
        status=doc["status"],
        created_at=doc.get("created_at")
    )


def _validate_doc_ready(doc_id: str):
    doc_res = supabase.table("documents").select("status").eq("id", doc_id).execute()
    if not doc_res.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{doc_id}' not found.")
    doc_status = doc_res.data[0]["status"]
    if doc_status != "COMPLETED":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document status is '{doc_status}'. Queries require 'COMPLETED'."
        )


@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    tags=["RAG Inference"]
)
async def chat_with_document(request: ChatRequest):
    """Synchronous inference endpoint (v1 backward compatibility)."""
    _validate_doc_ready(request.document_id)
    try:
        return answer_question(doc_id=request.document_id, query=request.question)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference error: {str(e)}"
        )


@app.post(
    "/api/v1/chat/stream",
    tags=["RAG Inference"]
)
async def chat_with_document_stream(request: ChatRequest):
    """
    Server-Sent Events (SSE) streaming endpoint.
    Emits real-time tokens, citation metadata, and termination signal.
    """
    _validate_doc_ready(request.document_id)
    return StreamingResponse(
        stream_answer_question(doc_id=request.document_id, query=request.question),
        media_type="text/event-stream"
    )


@app.get(
    "/api/v1/documents/{doc_id}/summarize",
    response_model=DocumentSummary,
    tags=["Agent Intelligence"]
)
async def summarize_document(doc_id: str):
    """
    Extracts structured executive summary with key takeaways and risk flags
    guaranteed by Pydantic response schema.
    """
    _validate_doc_ready(doc_id)
    try:
        return generate_document_summary(doc_id=doc_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate structured summary: {str(e)}"
        )