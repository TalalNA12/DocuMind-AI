import os
from pathlib import Path
from celery import Celery
from dotenv import load_dotenv
from services.ingest import process_and_store_document

# 1. Load environment variables from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(os.path.join(BASE_DIR, ".env"))

BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# 2. Celery Application Instance Definition
celery_app = Celery(
    "documind_worker",
    broker=BROKER_URL,
    backend=RESULT_BACKEND
)

# 3. Unified Operational & SSL Settings
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,             # 5 minutes max per task
    task_soft_time_limit=240,         # Soft timeout at 4 minutes
    worker_prefetch_multiplier=1,     # Fair distribution
    worker_concurrency=2,
    broker_use_ssl={"ssl_cert_reqs": None} if BROKER_URL.startswith("rediss://") else False,
    redis_backend_use_ssl={"ssl_cert_reqs": None} if RESULT_BACKEND.startswith("rediss://") else False,
)

# 4. Background Asynchronous Ingestion Task
@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    autoretry_for=(Exception,),
    name="tasks.process_document_task"
)
def process_document_task(self, doc_id: str, file_path: str):
    """
    Asynchronous Celery Task:
    Wraps heavy PDF ingestion, embedding generation, and DB insertion.
    """
    try:
        print(f"[CELERY WORKER] Received Task ID: {self.request.id} for Doc ID: {doc_id}")
        print(f"[CELERY WORKER] Processing file: {file_path} (Attempt {self.request.retries + 1}/3)")
        
        process_and_store_document(doc_id=doc_id, file_path=file_path)
        
        print(f"[CELERY WORKER] Successfully completed task {self.request.id} for Doc ID: {doc_id}")
        return {
            "status": "SUCCESS",
            "doc_id": doc_id,
            "task_id": self.request.id
        }

    except Exception as exc:
        print(f"[CELERY WORKER ERROR] Task {self.request.id} failed: {str(exc)}")
        raise exc