from __future__ import annotations

from fastapi.templating import Jinja2Templates

from app_config import BASE_DIR


template_engine = Jinja2Templates(directory=str(BASE_DIR / "templates"))
