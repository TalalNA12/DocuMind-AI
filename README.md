# DocuMind AI — Grounded Document Intelligence Engine

DocuMind AI is a production-grade, asynchronous Retrieval-Augmented Generation (RAG) platform designed to eliminate hallucinations and provide mathematically verifiable source attribution for document intelligence.

---

## ⚡ Key Highlights & Architecture

* **Asymmetric Latent Space Alignment:** Utilizes Google's 768-dimensional normalized embedding model with task-conditioned vectors (`RETRIEVAL_DOCUMENT` vs. `RETRIEVAL_QUERY`) to align user intent with target document knowledge on a shared hypersphere.
* **In-Database HNSW Vector Search:** Leverages PostgreSQL `pgvector` with Hierarchical Navigable Small World (HNSW) graph indexing and custom RPC procedures to achieve sub-millisecond $O(\log N)$ retrieval without loading embeddings into Python memory.
* **Distributed Non-Blocking Task Pipeline:** Implements Celery and Redis to handle PDF parsing, sliding-window chunking, and batch vector generation asynchronously with immediate `HTTP 202 Accepted` response cycles.
* **Multi-Tiered Anti-Hallucination Guardrails:** Enforces pre-inference cosine similarity gating ($\ge 0.50$), low-temperature generation ($T = 0.2$), and granular chunk provenance tracking.
* **Live Provenance Inspector:** Full-stack Next.js dashboard that maps every generated claim to its exact database chunk ID, similarity match percentage, and raw source text.

---

## 🛠️ Tech Stack

* **Frontend:** Next.js (App Router), TypeScript, Tailwind CSS, Lucide Icons, React Markdown
* **Backend Gateway:** FastAPI, Pydantic v2, Uvicorn
* **Task Queue & Broker:** Celery, Redis
* **Database & Vector Store:** Supabase (PostgreSQL 15+), pgvector (HNSW Indexing)
* **LLM & Embedding Models:** Google Gemini 3.6 Flash, `gemini-embedding-001`

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Ensure you have **Python 3.10+**, **Node.js 18+**, and **Redis** installed and running.

### 2. Backend Setup
```bash
# Clone the repository
git clone [https://github.com/](https://github.com/)<YOUR_USERNAME>/documind-ai.git
cd documind-ai

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your Gemini, Supabase, and Redis credentials
3. Launch Services
Terminal 1 — Celery Background Worker:

Bash
source venv/bin/activate
celery -A services.tasks.celery_app worker --loglevel=info
Terminal 2 — FastAPI Application Gateway:

Bash
source venv/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
Terminal 3 — Next.js Web Dashboard:

Bash
cd frontend
npm install
npm run dev
Open http://localhost:3000 to access the live dashboard or http://localhost:8000/docs for the interactive OpenAPI specifications.