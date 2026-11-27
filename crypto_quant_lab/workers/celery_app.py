"""Celery application configuration for distributed research & download workers."""

from __future__ import annotations

from celery import Celery

from crypto_quant_lab.app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "crypto_quant_lab",
    broker=settings.celery_broker_url or "redis://localhost:6379/0",
    backend=settings.celery_result_backend or "redis://localhost:6379/1",
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(name="cql.run_research_job")
def run_research_job_celery(job_id: int) -> dict[str, str]:
    from crypto_quant_lab.workers.job_runner import execute_job_by_id

    execute_job_by_id(job_id)
    return {"status": "completed", "job_id": str(job_id)}
