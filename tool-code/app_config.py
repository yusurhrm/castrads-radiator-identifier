from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
VISION_UPLOADS_DIR = UPLOADS_DIR / "vision"
LOGS_DIR = BASE_DIR / "logs"
VISION_LOG_FILE = LOGS_DIR / "vision_debug.log"
ACTIVE_MODEL_FILE = BASE_DIR / "active_model.json"
MISSING = "MISSING"
DEFAULT_CONFIDENCE_THRESHOLD = 0.86

# Optional local Qwen/DashScope API configuration.
# You can set the key directly for local/offline distribution, or leave it empty
# and use DASHSCOPE_API_KEY / QWEN_API_KEY from the environment instead.
QWEN_API_KEY = "sk-171a9201571b4117b011b698b9c49634"
QWEN_API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_VISION_MODEL = "qwen3.6-plus"
QWEN_MAX_VISION_IMAGES = 250
QWEN_MAX_IMAGE_BASE64_BYTES = 10 * 1024 * 1024
QWEN_MAX_IMAGE_PIXELS = 2_000_000
QWEN_MIN_IMAGE_SIDE = 11
QWEN_MAX_IMAGE_ASPECT_RATIO = 200


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    UPLOADS_DIR.mkdir(exist_ok=True)
    VISION_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)


def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
