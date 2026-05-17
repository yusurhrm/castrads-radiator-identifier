from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from decision_tree import tree_to_flowchart
from model_store import get_active_model_id, load_active_model, load_model
from web_templates import template_engine


router = APIRouter()


@router.get("/flowchart", response_class=HTMLResponse)
def flowchart(request: Request):
    model = load_active_model()
    if model is None:
        return HTMLResponse("<h2>No active model. Please train and activate one in /admin.</h2>")
    return render_flowchart(request, model)


@router.get("/admin/model/{model_id}/flowchart", response_class=HTMLResponse)
def model_flowchart(request: Request, model_id: str):
    model = load_model(model_id)
    if model is None:
        return HTMLResponse("<h2>Model not found.</h2>", status_code=404)
    return render_flowchart(request, model)


def render_flowchart(request: Request, model: dict[str, Any]) -> HTMLResponse:
    nodes, edges = tree_to_flowchart(model["tree"])
    return template_engine.TemplateResponse(
        request=request,
        name="flowchart.html",
        context={
            "nodes": nodes,
            "edges": edges,
            "model": model["metadata"],
            "is_active": model["id"] == get_active_model_id(),
        },
    )
