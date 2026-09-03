import os
import io
import re
import logging
from typing import List, Dict, Any
from google import genai
from google.genai import types
from pypdf import PdfReader
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


def extract_text_from_pdf(file_path: str) -> str:
    """Reads a local PDF file and extracts clean text with page boundary markers."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at path: {file_path}")

    reader = PdfReader(file_path)
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
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

        if not full_text.strip():
            raise ValueError("No extractable text found in PDF.")

        # 3. Chunking logic (keep your existing chunking implementation)
        chunks = chunk_text(full_text) # or whatever your chunking function is named

        # 4. Embed & Store in document_chunks
        for idx, chunk in enumerate(chunks):
            # Your existing embedding generation and insertion logic:
            embedding = generate_embedding(chunk)
            supabase.table("document_chunks").insert({
                "document_id": doc_id,
                "chunk_index": idx,
                "content": chunk,
                "embedding": embedding
            }).execute()

        # 5. Mark document as COMPLETED
        supabase.table("documents").update({"status": "COMPLETED"}).eq("id", doc_id).execute()
        print(f"[+] Document {doc_id} successfully indexed.")
        return {"chunks_indexed": len(chunks)}

    except Exception as e:
        print(f"[ERROR] Pipeline execution failed for Doc ID {doc_id}: {e}")
        supabase.table("documents").update({"status": "ERROR"}).eq("id", doc_id).execute()
        raise e