import os
import ssl
from pathlib import Path
from celery import Celery
from dotenv import load_dotenv
from services.ingest import process_and_store_document

# 1. Load local environment variables if available
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(os.path.join(BASE_DIR, ".env"))

# 2. Extract Redis connection string with fallbacks
REDIS_URL = (
    os.getenv("REDIS_URL")
    or os.getenv("CELERY_BROKER_URL")
    or "redis://localhost:6379/0"
)

# 3. Instantiate Celery
celery_app = Celery(
    "tasks",
    broker=REDIS_URL,
    backend=REDIS_URL
)

# 4. Configure SSL bypass for Upstash rediss:// endpoints
if REDIS_URL.startswith("rediss://"):
    celery_app.conf.update(
        broker_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
        redis_backend_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE}
    )

# 5. Core Celery settings
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    broker_connection_retry_on_startup=True
)

# 6. Registered Ingestion Task
@celery_app.task(bind=True, name="services.tasks.process_document_task")
def process_document_task(self, doc_id: str, file_path: str):
    """
    Background worker task to parse, chunk, embed, and index documents.
    """
    try:
        result = process_and_store_document(doc_id, file_path)
        return {"status": "SUCCESS", "document_id": doc_id, "result": result}
    except Exception as exc:
        self.retry(exc=exc, countdown=5, max_retries=3)