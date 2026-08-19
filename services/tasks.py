import os
from celery import Celery
from dotenv import load_dotenv
from services.ingest import process_and_store_document

# 1. Load environment variables
load_dotenv()

# 2. Redis Connection URL Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# 3. Celery Application Instance Definition
celery_app = Celery(
    "documind_worker",
    broker=REDIS_URL,       # Message Broker: Redis queue where tasks wait
    backend=REDIS_URL      # Result Backend: Stores task state & return values
)

# 4. Celery Worker Optimization & Operational Settings
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,          # Marks state as 'STARTED' immediately
    task_time_limit=300,              # Hard timeout: 5 minutes max per task
    task_soft_time_limit=240,         # Soft timeout: Throws exception at 4 mins to clean up
    worker_prefetch_multiplier=1,      # Fair distribution: Worker only grabs 1 job at a time
    worker_concurrency=2              # Number of concurrent worker threads/processes
)

# 5. Background Asynchronous Ingestion Task
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
    Wraps the heavy PDF ingestion, embedding generation, and DB insertion.
    
    Parameters:
    - self: The bound task instance (enables self.request, retries, and state tracking).
    - doc_id: UUID of the document in Supabase.
    - file_path: Local filesystem path to the uploaded PDF.
    """
    try:
        print(f"[CELERY WORKER] Received Task ID: {self.request.id} for Doc ID: {doc_id}")
        print(f"[CELERY WORKER] Processing file: {file_path} (Attempt {self.request.retries + 1}/3)")
        
        # Execute the core synchronous pipeline we built in ingest.py
        process_and_store_document(doc_id=doc_id, file_path=file_path)
        
        print(f"[CELERY WORKER] Successfully completed task {self.request.id} for Doc ID: {doc_id}")
        return {
            "status": "SUCCESS",
            "doc_id": doc_id,
            "task_id": self.request.id
        }

    except Exception as exc:
        print(f"[CELERY WORKER ERROR] Task {self.request.id} failed: {str(exc)}")
        # Let Celery's autoretry mechanism handle exponential backoff up to max_retries
        raise exc