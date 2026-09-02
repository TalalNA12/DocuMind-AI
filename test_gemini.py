import os, time
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment.")

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=45.0)
)

print("[1/2] Testing Embedding API...")
t0 = time.time()
try:
    emb = client.models.embed_content(
        model="gemini-embedding-001",
        contents="What are security principles?",
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768
        )
    )
    print(f"  -> Embedding OK in {round(time.time() - t0, 2)}s (dims: {len(emb.embeddings[0].values)})")
except Exception as e:
    print(f"  -> Embedding FAILED: {e}")

print("[2/2] Testing Generation API...")
t1 = time.time()
try:
    res = client.models.generate_content(
        model="gemini-3.5-flash",
        contents="Respond with only: DOCUMIND_ONLINE"
    )
    print(f"  -> Generation OK in {round(time.time() - t1, 2)}s: {res.text.strip()}")
except Exception as e:
    print(f"  -> Generation FAILED: {e}")
