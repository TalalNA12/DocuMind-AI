import os
import json
from typing import List, Dict, Any, AsyncGenerator
from dotenv import load_dotenv
from supabase import create_client, Client
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

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
genai_client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(timeout=30.0)
)

# --- Constants ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
GENERATION_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
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


def generate_query_embedding(text: str) -> List[float]:
    """
    Generates a 768-dimensional embedding vector for query retrieval.
    """
    try:
        response = genai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768
            )
        )
        return response.embeddings[0].values
    except Exception as e:
        raise RuntimeError(f"Failed to generate query embedding: {e}") from e


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
            "document_id": doc_id,
            "query_embedding": query_vector,
            "match_count": match_count
        }
        response = supabase.rpc("match_document_chunks", rpc_params).execute()
        return response.data or []
    except Exception as e:
        raise RuntimeError(f"Vector search failed in Supabase RPC: {e}") from e


def answer_question(doc_id: str, query: str) -> Dict[str, Any]:
    """
    Standard synchronous RAG pipeline execution.
    Returns: { 'answer': str, 'confidence_score': float, 'citations': list }
    """
    # 1. Embed query
    query_vector = generate_query_embedding(query)

    # 2. Retrieve top-k chunks
    chunks = retrieve_relevant_chunks(doc_id, query_vector, match_count=DEFAULT_MATCH_COUNT)

    if not chunks:
        return {
            "answer": "No indexed content was found for this document.",
            "confidence_score": 0.0,
            "citations": []
        }

    top_score = float(chunks[0].get("similarity", 0.0))

    # 3. Guardrail: Confidence Floor
    if top_score < CONFIDENCE_THRESHOLD:
        return {
            "answer": "I could not find sufficient information in the document to reliably answer this question.",
            "confidence_score": round(top_score, 4),
            "citations": []
        }

    # 4. Format Context and Source Citations
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

    # 5. Synthesis
    try:
        response = genai_client.models.generate_content(
            model=GENERATION_MODEL,
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
        raise RuntimeError(f"Synthesis failed in Gemini generation: {e}") from e


async def stream_answer_question(doc_id: str, query: str) -> AsyncGenerator[str, None]:
    """
    Asynchronous SSE streaming generator for real-time frontend token delivery.
    Yields data lines formatted as: data: {"type": ..., ...}\n\n
    """
    # 1. Embed query
    try:
        query_vector = generate_query_embedding(query)
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        return

    # 2. Retrieve chunks
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

    # 3. Confidence Threshold Check
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

    # 4. Prepare Context and Metadata
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

    # 5. Emit Initial Metadata Frame (Citations & Score)
    meta_frame = {
        "type": "metadata",
        "confidence_score": round(top_score, 4),
        "citations": citations
    }
    yield f"data: {json.dumps(meta_frame)}\n\n"

    # 6. Stream Generation Tokens
    try:
        response_stream = genai_client.models.generate_content_stream(
            model=GENERATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2
            )
        )
        for chunk in response_stream:
            if chunk.text:
                token_frame = {"type": "token", "text": chunk.text}
                yield f"data: {json.dumps(token_frame)}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'done': True})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"


class DocumentSummary(BaseModel):
    document_id: str = Field(..., description="Unique UUID of the document")
    summary: str = Field(..., description="Executive summary of the document")
    key_takeaways: List[str] = Field(default_factory=list, description="Extracted bullet points or key takeaways")

def generate_document_summary(doc_id: str) -> Dict[str, Any]:
    """
    Retrieves the initial chunks of the document and generates
    an executive summary and key takeaways.
    """
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

        res = genai_client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are an elite technical document analyst. Summarize clearly and objectively.",
                temperature=0.2
            )
        )

        raw_text = res.text.strip() if res.text else ""

        # Parse out summary vs takeaways if present, or pass cleanly
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        takeaways = [l.lstrip("*-•123456789. ") for l in lines if l.startswith(("*", "-", "•")) or (len(l) > 2 and l[0].isdigit() and l[1] in ". ")]

        return {
            "document_id": doc_id,
            "summary": raw_text,
            "key_takeaways": takeaways
        }
    except Exception as e:
        raise RuntimeError(f"Summary generation failed: {e}") from e