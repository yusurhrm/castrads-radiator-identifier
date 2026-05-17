from __future__ import annotations

from fastapi import Request
from fastapi.responses import HTMLResponse

from app_config import DEFAULT_CONFIDENCE_THRESHOLD
from flow_runtime import get_next_step_context
from model_store import load_active_model
from web_templates import template_engine


def render_next_step(request: Request) -> HTMLResponse:
    model = load_active_model()
    if model is None:
        return HTMLResponse("<h2>No active model. Please train and activate one in /admin.</h2>")

    threshold = float(model["config"].get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))
    status, context = get_next_step_context(model, threshold)
    if status == "question":
        return template_engine.TemplateResponse(
            request=request,
            name="question.html",
            context=context,
        )
    if status in ("result", "ambiguous"):
        return template_engine.TemplateResponse(
            request=request,
            name="result.html",
            context=context,
        )
    if status == "no_match":
        return HTMLResponse("<h2>No matching candidates.</h2>")
    return HTMLResponse("<h2>No decision node is available. Please restart identification.</h2>")
