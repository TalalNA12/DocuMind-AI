import os
from typing import List, Dict, Any
from google import genai
from google.genai import types
from supabase import create_client, Client
from dotenv import load_dotenv
import logging

# Suppress SDK informational warnings from google.genai
logging.getLogger("google.genai").setLevel(logging.ERROR)

# 1. Environment & Global Client Initialization
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not all([GEMINI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY]):
    raise EnvironmentError("Missing critical environment variables for RAG service.")

genai_client = genai.Client(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Confidence threshold: Below 50% match, trigger early exit against hallucinations
CONFIDENCE_THRESHOLD = 0.50




def generate_query_embedding(query_text: str) -> List[float]:
    """
    Vectorizes the user's question using task_type='RETRIEVAL_QUERY'.
    Outputs a 768-dimensional L2-normalized vector.
    """
    response = genai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=query_text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768
        )
    )
    return response.embeddings[0].values


def retrieve_relevant_chunks(doc_id: str, query_embedding: List[float], match_count: int = 4) -> List[Dict[str, Any]]:
    """
    Executes the PostgreSQL stored procedure `match_document_chunks` via Supabase RPC.
    Utilizes the HNSW cosine graph index for O(log N) retrieval.
    """
    rpc_params = {
        "query_embedding": query_embedding,
        "filter_document_id": doc_id,    # <--- matched to SQL parameter name
        "match_count": match_count
    }
    
    response = supabase.rpc("match_document_chunks", rpc_params).execute()
    return response.data or []


def generate_answer_with_citations(query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Synthesizes a grounded answer using Gemini 3.6 Flash, enforcing strict citation markers.
    Includes automated retry with backoff against temporary capacity spikes.
    """
    # 1. Anti-hallucination guardrail check
    top_score = chunks[0]["similarity"] if chunks else 0.0
    if top_score < CONFIDENCE_THRESHOLD:
        return {
            "answer": "I could not find sufficient information in the document to reliably answer this question.",
            "confidence_score": round(top_score, 4),
            "citations": []
        }

    # 2. Build structured context and citation payload
    context_blocks = []
    citations = []
    for idx, c in enumerate(chunks):
        source_num = idx + 1
        similarity_pct = round(c["similarity"] * 100, 1)
        context_blocks.append(f"[Source {source_num} (Relevance: {similarity_pct}%)]:\n{c['content']}")
        citations.append({
            "source_id": source_num,
            "chunk_id": c["id"],
            "similarity": c["similarity"],
            "preview": c["content"][:120] + "..."
        })

    context_str = "\n\n".join(context_blocks)

    system_instruction = (
        "You are DocuMind AI, an elite document intelligence analyst. "
        "Answer the user's question strictly and exclusively using the provided context chunks. "
        "If the answer cannot be deduced from the context, state clearly that the document lacks the information. "
        "Always cite your sources using [Source X] notation whenever referencing a point."
    )

    prompt = f"### CONTEXT FROM DOCUMENT:\n{context_str}\n\n### USER QUESTION:\n{query}"

    # 3. Model generation with retry & fallback
    models_to_try = ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.5-flash"]
    last_err = None

    for model_name in models_to_try:
        for attempt in range(2):
            try:
                response = genai_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.2
                    )
                )
                return {
                    "answer": response.text,
                    "confidence_score": round(top_score, 4),
                    "citations": citations
                }
            except Exception as e:
                last_err = e
                time.sleep(1)  # brief pause before next attempt

    raise last_err


def answer_question(doc_id: str, query: str) -> Dict[str, Any]:
    """Top-level orchestrator for the RAG inference pipeline."""
    # Step A: Vectorize question
    query_vector = generate_query_embedding(query)
    
    # Step B: Fetch Top-K nearest chunks from PostgreSQL
    chunks = retrieve_relevant_chunks(doc_id, query_vector, match_count=4)
    
    if not chunks:
        return {
            "answer": "No indexed content found for this document ID.",
            "confidence_score": 0.0,
            "citations": []
        }

    # Step C: Synthesize grounded response
    return generate_answer_with_citations(query, chunks)