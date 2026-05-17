from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from app_config import now_id
from vision_image_limits import prepare_vision_uploads


vision_tasks: dict[str, dict[str, Any]] = {}


def create_vision_task(
    model_id: str,
    image_paths: list[Path],
    warnings: list[str],
    mode: str,
) -> str:
    task_id = now_id()
    vision_tasks[task_id] = {
        "model_id": model_id,
        "image_paths": image_paths,
        "warnings": warnings,
        "mode": mode,
    }
    return task_id


def get_vision_task(task_id: str) -> dict[str, Any] | None:
    return vision_tasks.get(task_id)


def pop_vision_task(task_id: str) -> dict[str, Any] | None:
    return vision_tasks.pop(task_id, None)


def save_user_vision_images(uploads: list[UploadFile]) -> tuple[list[Path], list[str]]:
    return prepare_vision_uploads(uploads)


def sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
