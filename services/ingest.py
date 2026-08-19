import os
import re
from typing import List, Dict, Any
from google import genai
from google.genai import types
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import create_client, Client
from dotenv import load_dotenv

# 1. Environment & Global Client Configurations
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not all([GEMINI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY]):
    raise EnvironmentError("Missing critical environment variables for Ingestion service.")

# Initialize Google GenAI & Supabase Clients
genai_client = genai.Client(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


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
        
        # Clean null bytes and collapse whitespace
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


def generate_gemini_embeddings(chunks: List[str], batch_size: int = 16) -> List[List[float]]:
    """
    Generates 768-dimensional embeddings using gemini-embedding-001.
    Sets task_type to RETRIEVAL_DOCUMENT for asymmetrical target indexing.
    """
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


def process_and_store_document(doc_id: str, file_path: str):
    """Parses, chunks, embeds, and batch-inserts the document into Supabase."""
    try:
        print(f"[*] Starting ingestion for Document ID: {doc_id}")
        
        # 1. Text extraction
        raw_text = extract_text_from_pdf(file_path)
        if not raw_text.strip():
            raise ValueError("PDF contains no machine-readable text.")

        # 2. Semantic chunking
        chunks = chunk_text(raw_text)
        print(f"[*] Created {len(chunks)} text chunks.")

        # 3. Vector embedding generation
        embeddings = generate_gemini_embeddings(chunks)
        print(f"[*] Generated {len(embeddings)} 768-dim embeddings via Gemini.")

        # 4. Construct SQL records payload
        records: List[Dict[str, Any]] = [
            {
                "document_id": doc_id,
                "content": chunk,
                "chunk_index": idx,
                "embedding": emb
            }
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]

        # 5. Batch insert into Supabase pgvector
        supabase.table("document_chunks").insert(records).execute()
        print(f"[*] Inserted {len(records)} chunk records into Supabase.")

        # 6. Mark status as COMPLETED
        supabase.table("documents").update({"status": "COMPLETED"}).eq("id", doc_id).execute()
        print(f"[SUCCESS] Document {doc_id} indexed in pgvector successfully.")

    except Exception as e:
        print(f"[ERROR] Pipeline execution failed for Doc ID {doc_id}: {str(e)}")
        supabase.table("documents").update({"status": "FAILED"}).eq("id", doc_id).execute()
        raise e