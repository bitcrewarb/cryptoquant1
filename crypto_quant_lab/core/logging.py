"""Structured application logging with optional database persistence."""

from __future__ import annotations

import logging
import sys
from typing import Any


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logger for structured console output."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger("crypto_quant_lab")
    root_logger.setLevel(numeric_level)
    if not root_logger.handlers:
        root_logger.addHandler(handler)
    root_logger.propagate = False
    return root_logger


def log_system_event(
    db_session: Any,
    level: str,
    module: str,
    message: str,
    context: dict[str, Any] | None = None,
) -> None:
    """Persist an operational log record to system_logs without leaking secrets."""
    from crypto_quant_lab.database.models import SystemLog

    try:
        entry = SystemLog(
            level=level.upper()[:20],
            module=module[:120],
            message=message[:2000],
            context_json=context or {},
        )
        db_session.add(entry)
        db_session.commit()
    except Exception:
        db_session.rollback()
