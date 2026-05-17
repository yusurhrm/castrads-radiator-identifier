from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from app_logging import log_vision, timed_vision
from flow_runtime import apply_answer, can_answer, go_back, skip_feature, start_session
from flow_views import render_next_step
from model_store import load_active_model
from vision_workflow import create_vision_task, save_user_vision_images
from web_templates import template_engine


router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def start(request: Request):
    model = load_active_model()
    return template_engine.TemplateResponse(
        request=request,
        name="start.html",
        context={"model": model["metadata"] if model else None},
    )


@router.post("/start", response_class=HTMLResponse)
def start_flow(request: Request, images: list[UploadFile] | None = File(default=None)):
    model = load_active_model()
    if model is None:
        return HTMLResponse("<h2>No active model. Please train and activate one in /admin.</h2>")

    upload_count = len([image for image in images or [] if image.filename])
    log_vision("start_flow.received", model_id=model["id"], upload_count=upload_count)
    with timed_vision("start_flow.prepare_uploads", upload_count=upload_count):
        image_paths, image_warnings = save_user_vision_images(images or [])
    log_vision(
        "start_flow.uploads_prepared",
        prepared_count=len(image_paths),
        warnings=image_warnings,
        image_paths=[str(path) for path in image_paths],
    )
    if image_paths:
        task_id = create_vision_task(model["id"], image_paths, image_warnings, "new_session")
        log_vision("start_flow.vision_task_created", task_id=task_id, model_id=model["id"])
        return template_engine.TemplateResponse(
            request=request,
            name="vision_loading.html",
            context={
                "task_id": task_id,
                "model": model["metadata"],
                "image_count": len(image_paths),
                "warnings": image_warnings,
                "mode": "new_session",
            },
        )

    start_session(model, {}, "", " ".join(image_warnings))
    log_vision("start_flow.no_images_start_session", warnings=image_warnings)
    return render_next_step(request)


@router.post("/answer", response_class=HTMLResponse)
def answer(request: Request, feature: str = Form(...), value: str = Form(...)):
    if not can_answer(feature):
        return RedirectResponse("/", status_code=303)
    apply_answer(feature, value)
    return render_next_step(request)


@router.post("/skip", response_class=HTMLResponse)
def skip(request: Request, feature: str = Form(...)):
    model = load_active_model()
    if model is None:
        return RedirectResponse("/", status_code=303)
    skip_feature(feature, model)
    return render_next_step(request)


@router.post("/back", response_class=HTMLResponse)
def back(request: Request):
    model = load_active_model()
    if model is None:
        return RedirectResponse("/", status_code=303)
    go_back(model)
    return render_next_step(request)
