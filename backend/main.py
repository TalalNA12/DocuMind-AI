import os
import shutil
import uuid
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Internal Service Imports
from services.ingest import supabase
from services.tasks import process_document_task
from services.rag import answer_question

# 1. App Initialization & Metadata
app = FastAPI(
    title="DocuMind AI",
    description="Production-grade asynchronous RAG engine powered by pgvector, Celery, and Gemini.",
    version="1.0.0"
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
    """Returns engine health status."""
    return {"status": "HEALTHY", "engine": "DocuMind AI"}


@app.post(
    "/api/v1/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Documents"]
)
async def upload_document(file: UploadFile = File(...)):
    """
    Accepts a PDF file upload, stores it locally, writes a 'PROCESSING' record
    to Supabase, and dispatches the ingestion pipeline to Celery.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Only PDF documents are accepted."
        )

    doc_id = str(uuid.uuid4())
    safe_filename = f"{doc_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    try:
        # Stream file to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Register document state in Supabase
        supabase.table("documents").insert({
            "id": doc_id,
            "filename": file.filename,
            "status": "PROCESSING"
        }).execute()

        # Dispatch async task to Redis / Celery
        process_document_task.delay(doc_id=doc_id, file_path=file_path)

        return UploadResponse(
            document_id=doc_id,
            filename=file.filename,
            status="PROCESSING",
            message="Document received and queued for asynchronous vector ingestion."
        )

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
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
    """
    Retrieves the current indexing state ('PROCESSING', 'COMPLETED', 'FAILED')
    for a given document ID.
    """
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


@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    tags=["RAG Inference"]
)
async def chat_with_document(request: ChatRequest):
    """
    Performs vector similarity search and cited LLM synthesis
    against a fully ingested document.
    """
    # Verify document existence and state
    doc_res = supabase.table("documents").select("status").eq("id", request.document_id).execute()
    if not doc_res.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{request.document_id}' not found."
        )

    doc_status = doc_res.data[0]["status"]
    if doc_status != "COMPLETED":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document is currently in '{doc_status}' state. Queries require 'COMPLETED' status."
        )

    # Execute RAG pipeline with graceful upstream exception handling
    try:
        result = answer_question(doc_id=request.document_id, query=request.question)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference provider temporarily unavailable: {str(e)}"
        )