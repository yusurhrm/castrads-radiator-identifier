from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from typing import Any

from app_config import VISION_LOG_FILE, ensure_dirs


def get_vision_logger() -> logging.Logger:
    ensure_dirs()
    logger = logging.getLogger("vision_debug")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(VISION_LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_vision(event: str, **payload: Any) -> None:
    safe_payload = {key: redact(value) for key, value in payload.items()}
    get_vision_logger().info(
        "%s %s",
        event,
        json.dumps(safe_payload, ensure_ascii=False, default=str),
    )


@contextmanager
def timed_vision(event: str, **payload: Any):
    start = time.perf_counter()
    log_vision(f"{event}.start", **payload)
    try:
        yield
    except Exception as exc:
        log_vision(f"{event}.error", elapsed_ms=elapsed_ms(start), error=str(exc), **payload)
        raise
    else:
        log_vision(f"{event}.end", elapsed_ms=elapsed_ms(start), **payload)


def elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def redact(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("sk-"):
        return value[:6] + "***"
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value
