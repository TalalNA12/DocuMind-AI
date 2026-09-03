import os
import ssl
from pathlib import Path
from celery import Celery
from dotenv import load_dotenv
from services.ingest import process_and_store_document

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(os.path.join(BASE_DIR, ".env"))

REDIS_URL = (
    os.getenv("REDIS_URL")
    or os.getenv("CELERY_BROKER_URL")
    or "redis://localhost:6379/0"
)

celery_app = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)

if REDIS_URL.startswith("rediss://"):
    celery_app.conf.update(
        broker_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
        redis_backend_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE}
    )

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    broker_connection_retry_on_startup=True,
    broker_transport_options={
        "socket_keepalive": True,
        "socket_timeout": 30,
        "retry_on_timeout": True
    },
    redis_backend_transport_options={
        "socket_keepalive": True,
        "socket_timeout": 30,
        "retry_on_timeout": True
    }
)

@celery_app.task(bind=True, name="services.tasks.process_document_task")
def process_document_task(self, doc_id: str, file_path: str):
    try:
        result = process_and_store_document(doc_id, file_path)
        return {"status": "SUCCESS", "document_id": doc_id, "result": result}
    except Exception as exc:
        self.retry(exc=exc, countdown=5, max_retries=3)