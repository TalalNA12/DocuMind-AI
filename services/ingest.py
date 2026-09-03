import os
import io
import re
import logging
from typing import List, Dict, Any
import pypdf
from pypdf import PdfReader
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import create_client, Client
from dotenv import load_dotenv

# Optional LangSmith tracing integration
try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not all([GEMINI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY]):
    raise EnvironmentError("Missing critical environment variables for Ingestion service.")

genai_client = genai.Client(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

logger = logging.getLogger("documind.ingest")
logger.setLevel(logging.INFO)


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Reads PDF bytes directly from memory and extracts cleaned text with page markers."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    extracted_pages = []

    for page_idx, page in enumerate(reader.pages):
        raw_page_text = page.extract_text()
        if not raw_page_text:
            continue
        
        cleaned_text = re.sub(r"\x00", "", raw_page_text)
        cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text).strip()

        if cleaned_text:
            extracted_pages.append(f"--- [Page {page_idx + 1}] ---\n{cleaned_text}")

    return "\n\n".join(extracted_pages)


def chunk_text(text: str) -> List[str]:
    """Splits text into 1000-character chunks with a 200-character sliding overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


@traceable(name="gemini_embed_chunks", run_type="embedding")
def generate_gemini_embeddings(chunks: List[str], batch_size: int = 16) -> List[List[float]]:
    """Generates 768-dimensional embeddings using gemini-embedding-001."""
    all_embeddings: List[List[float]] = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        
        response = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768
            )
        )
        
        for item in response.embeddings:
            all_embeddings.append(item.values)

    return all_embeddings


@traceable(name="documind_ingest_pipeline", run_type="chain")
def process_and_store_document(doc_id: str, file_path: str):
    print(f"[*] Starting ingestion for Document ID: {doc_id}")
    try:
        # 1. Acquire PDF bytes from Local Disk or Supabase Storage
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()
        else:
            # Download directly from Supabase Storage bucket 'documents'
            pdf_bytes = supabase.storage.from_("documents").download(file_path)

        # 2. Extract text from memory
        full_text = extract_text_from_pdf_bytes(pdf_bytes)

        if not full_text.strip():
            raise ValueError("No extractable text found in PDF.")

        # 3. Split into overlapping chunks
        chunks = chunk_text(full_text)
        if not chunks:
            raise ValueError("Text chunking resulted in 0 chunks.")

        # 4. Generate batch embeddings
        embeddings = generate_gemini_embeddings(chunks)

        # 5. Insert records into document_chunks
        records = [
            {
                "document_id": doc_id,
                "chunk_index": idx,
                "content": chunk,
                "embedding": emb
            }
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]

        # Batch insert to Supabase
        supabase.table("document_chunks").insert(records).execute()

        # 6. Update document state to COMPLETED
        supabase.table("documents").update({"status": "COMPLETED"}).eq("id", doc_id).execute()
        print(f"[+] Document {doc_id} successfully indexed ({len(chunks)} chunks).")
        return {"chunks_indexed": len(chunks)}

    except Exception as e:
        print(f"[ERROR] Pipeline execution failed for Doc ID {doc_id}: {e}")
        supabase.table("documents").update({"status": "ERROR"}).eq("id", doc_id).execute()
        raise e