import os
import json
import time
import logging
import asyncio
from typing import List, Dict, Any, AsyncGenerator
from dotenv import load_dotenv
from supabase import create_client, Client
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger("documind.rag")
logger.setLevel(logging.INFO)

# --- Configuration & Credentials ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is missing.")
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("Supabase credentials missing from environment.")

# --- Clients ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# --- Constants & Fallbacks ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
PRIMARY_GENERATION_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
FALLBACK_GENERATION_MODELS = [
    PRIMARY_GENERATION_MODEL,
    "gemini-3-flash",
    "gemini-2.5-flash"
]
CONFIDENCE_THRESHOLD = 0.50
DEFAULT_MATCH_COUNT = 4

SYSTEM_INSTRUCTION = (
    "You are DocuMind AI, an elite document intelligence analyst. "
    "Answer the user's question strictly and exclusively using the provided context chunks. "
    "If the answer cannot be deduced from the context, explicitly state: "
    "'I cannot find sufficient information in the provided document to answer this question.' "
    "Never hallucinate or extrapolate beyond what is documented. "
    "Always cite your sources using [Source X] notation whenever referencing facts or statements."
)


async def generate_query_embedding_async(text: str, max_retries: int = 3) -> List[float]:
    """
    Asynchronously generates a 768-dimensional embedding vector.
    Uses genai_client.aio to avoid synchronous socket contention.
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            response = await genai_client.aio.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=768
                )
            )
            return response.embeddings[0].values
        except Exception as e:
            last_exception = e
            logger.warning(f"Async embedding attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1.0)

    raise RuntimeError(f"Embedding service timed out: {last_exception}")


def retrieve_relevant_chunks(
    doc_id: str, 
    query_vector: List[float], 
    match_count: int = DEFAULT_MATCH_COUNT
) -> List[Dict[str, Any]]:
    """
    Executes an HNSW cosine similarity search via Supabase RPC match_document_chunks.
    """
    try:
        rpc_params = {
            "filter_document_id": doc_id,
            "query_embedding": query_vector,
            "match_count": match_count
        }
        response = supabase.rpc("match_document_chunks", rpc_params).execute()
        return response.data or []
    except Exception as e:
        raise RuntimeError(f"Vector search failed in Supabase RPC: {e}") from e


async def answer_question(doc_id: str, query: str) -> Dict[str, Any]:
    """
    Synchronous fallback endpoint using async query vector generation.
    Returns: { 'answer': str, 'confidence_score': float, 'citations': list }
    """
    query_vector = await generate_query_embedding_async(query)

    chunks = retrieve_relevant_chunks(doc_id, query_vector, match_count=DEFAULT_MATCH_COUNT)

    if not chunks:
        return {
            "answer": "No indexed content was found for this document.",
            "confidence_score": 0.0,
            "citations": []
        }

    top_score = float(chunks[0].get("similarity", 0.0))

    if top_score < CONFIDENCE_THRESHOLD:
        return {
            "answer": "I could not find sufficient information in the document to reliably answer this question.",
            "confidence_score": round(top_score, 4),
            "citations": []
        }

    context_blocks = []
    citations = []
    for idx, c in enumerate(chunks):
        source_idx = idx + 1
        similarity_pct = round(c.get("similarity", 0.0) * 100, 1)
        context_blocks.append(f"[Source {source_idx} (Relevance: {similarity_pct}%)]:\n{c.get('content', '')}")
        citations.append({
            "source_id": source_idx,
            "chunk_id": c.get("id"),
            "similarity": round(c.get("similarity", 0.0), 4),
            "preview": c.get("content", "")[:120] + "..."
        })

    context_str = "\n\n".join(context_blocks)
    prompt = f"### CONTEXT FROM DOCUMENT:\n{context_str}\n\n### USER QUESTION:\n{query}"

    for model_id in FALLBACK_GENERATION_MODELS:
        try:
            response = genai_client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.2
                )
            )
            return {
                "answer": response.text.strip() if response.text else "No response generated.",
                "confidence_score": round(top_score, 4),
                "citations": citations
            }
        except Exception as e:
            logger.warning(f"Synthesis failed on model {model_id}: {e}. Retrying fallback...")
            continue

    raise RuntimeError("Inference generation failed across all available Gemini model endpoints.")


async def stream_answer_question(doc_id: str, query: str) -> AsyncGenerator[str, None]:
    """
    Asynchronous SSE streaming generator for real-time frontend token delivery.
    Yields data lines formatted as: data: {"type": ..., ...}\n\n
    """
    try:
        query_vector = await generate_query_embedding_async(query)
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        return

    try:
        chunks = retrieve_relevant_chunks(doc_id, query_vector, match_count=DEFAULT_MATCH_COUNT)
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        return

    if not chunks:
        payload = {
            "type": "terminal",
            "answer": "No indexed content was found for this document.",
            "confidence_score": 0.0,
            "citations": [],
            "done": True
        }
        yield f"data: {json.dumps(payload)}\n\n"
        return

    top_score = float(chunks[0].get("similarity", 0.0))

    if top_score < CONFIDENCE_THRESHOLD:
        payload = {
            "type": "terminal",
            "answer": "I could not find sufficient information in the document to reliably answer this question.",
            "confidence_score": round(top_score, 4),
            "citations": [],
            "done": True
        }
        yield f"data: {json.dumps(payload)}\n\n"
        return

    context_blocks = []
    citations = []
    for idx, c in enumerate(chunks):
        source_idx = idx + 1
        similarity_pct = round(c.get("similarity", 0.0) * 100, 1)
        context_blocks.append(f"[Source {source_idx} (Relevance: {similarity_pct}%)]:\n{c.get('content', '')}")
        citations.append({
            "source_id": source_idx,
            "chunk_id": c.get("id"),
            "similarity": round(c.get("similarity", 0.0), 4),
            "preview": c.get("content", "")[:120] + "..."
        })

    context_str = "\n\n".join(context_blocks)
    prompt = f"### CONTEXT FROM DOCUMENT:\n{context_str}\n\n### USER QUESTION:\n{query}"

    # Emit metadata first
    meta_frame = {
        "type": "meta",
        "confidence_score": round(top_score, 4),
        "citations": citations
    }
    yield f"data: {json.dumps(meta_frame)}\n\n"

    # Stream tokens with model fallback cascade
    stream_successful = False
    for model_id in FALLBACK_GENERATION_MODELS:
        try:
            response_stream = genai_client.models.generate_content_stream(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.2
                )
            )
            for chunk in response_stream:
                if chunk.text:
                    token_frame = {"type": "token", "token": chunk.text}
                    yield f"data: {json.dumps(token_frame)}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'done': True})}\n\n"
            stream_successful = True
            break
        except Exception as e:
            logger.warning(f"Stream generation dropped on {model_id}: {e}. Trying fallback model...")
            continue

    if not stream_successful:
        yield f"data: {json.dumps({'type': 'error', 'error': 'Inference capacity exhausted across models. Please retry shortly.'})}\n\n"


class DocumentSummary(BaseModel):
    document_id: str = Field(..., description="Unique UUID of the document")
    summary: str = Field(..., description="Executive summary of the document")
    key_takeaways: List[str] = Field(default_factory=list, description="Extracted bullet points or key takeaways")

def generate_document_summary(doc_id: str) -> Dict[str, Any]:
    try:
        response = (
            supabase.table("document_chunks")
            .select("content, chunk_index")
            .eq("document_id", doc_id)
            .order("chunk_index", desc=False)
            .limit(6)
            .execute()
        )
        chunks = response.data or []

        if not chunks:
            return {
                "document_id": doc_id,
                "summary": "No content found for this document to summarize.",
                "key_takeaways": []
            }

        context = "\n\n".join([f"[Section {c['chunk_index']}]:\n{c['content']}" for c in chunks])
        prompt = (
            "Analyze the following document sections and produce:\n"
            "1. A concise executive summary (under 150 words).\n"
            "2. 3 to 5 key takeaways as bullet points.\n\n"
            f"### DOCUMENT CONTENT:\n{context}"
        )

        res = None
        for model_id in FALLBACK_GENERATION_MODELS:
            try:
                res = genai_client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction="You are an elite technical document analyst. Summarize clearly and objectively.",
                        temperature=0.2
                    )
                )
                break
            except Exception:
                continue

        raw_text = res.text.strip() if (res and res.text) else ""
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        takeaways = [l.lstrip("*-•123456789. ") for l in lines if l.startswith(("*", "-", "•")) or (len(l) > 2 and l[0].isdigit() and l[1] in ". ")]

        return {
            "document_id": doc_id,
            "summary": raw_text,
            "key_takeaways": takeaways
        }
    except Exception as e:
        raise RuntimeError(f"Summary generation failed: {e}") from e