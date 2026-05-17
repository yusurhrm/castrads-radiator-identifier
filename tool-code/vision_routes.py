from __future__ import annotations

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse

from app_logging import log_vision, timed_vision
from flow_runtime import start_session, update_vision_suggestions
from flow_views import render_next_step
from model_store import load_active_model, load_model
from vision_service import analyze_images_stream
from vision_workflow import (
    create_vision_task,
    get_vision_task,
    pop_vision_task,
    save_user_vision_images,
    sse_event,
)
from web_templates import template_engine


router = APIRouter(prefix="/vision")


@router.get("/upload", response_class=HTMLResponse)
def vision_upload_page(request: Request):
    model = load_active_model()
    if model is None:
        return HTMLResponse("<h2>No active model. Please train and activate one in /admin.</h2>")
    return template_engine.TemplateResponse(
        request=request,
        name="vision_upload.html",
        context={"model": model["metadata"]},
    )


@router.post("/start", response_class=HTMLResponse)
def start_inline_vision(request: Request, images: list[UploadFile] | None = File(default=None)):
    model = load_active_model()
    if model is None:
        return HTMLResponse("<h2>No active model. Please train and activate one in /admin.</h2>")

    upload_count = len([image for image in images or [] if image.filename])
    log_vision("inline_vision.received", model_id=model["id"], upload_count=upload_count)
    with timed_vision("inline_vision.prepare_uploads", upload_count=upload_count):
        image_paths, image_warnings = save_user_vision_images(images or [])
    if not image_paths:
        update_vision_suggestions({}, error=" ".join(image_warnings))
        return render_next_step(request)

    task_id = create_vision_task(model["id"], image_paths, image_warnings, "update_session")
    log_vision("inline_vision.task_created", task_id=task_id, model_id=model["id"])
    return template_engine.TemplateResponse(
        request=request,
        name="vision_loading.html",
        context={
            "task_id": task_id,
            "model": model["metadata"],
            "image_count": len(image_paths),
            "warnings": image_warnings,
            "mode": "update_session",
        },
    )


@router.get("/{task_id}/stream")
def stream_vision(task_id: str):
    log_vision("vision_stream.open", task_id=task_id)
    task = get_vision_task(task_id)
    if not task:
        log_vision("vision_stream.task_missing", task_id=task_id)
        return StreamingResponse(
            iter([sse_event({"type": "error", "error": "Image understanding task expired."})]),
            media_type="text/event-stream",
        )

    model = load_model(task["model_id"])
    if model is None:
        log_vision("vision_stream.model_missing", task_id=task_id, model_id=task["model_id"])
        return StreamingResponse(
            iter([sse_event({"type": "error", "error": "Model not found."})]),
            media_type="text/event-stream",
        )

    def event_stream():
        log_vision(
            "vision_stream.started",
            task_id=task_id,
            model_id=model["id"],
            image_count=len(task["image_paths"]),
            warnings=task.get("warnings", []),
        )
        for warning in task.get("warnings", []):
            yield sse_event({"type": "warning", "message": warning})
        for event in analyze_images_stream(model, task["image_paths"]):
            log_vision(
                "vision_stream.event",
                task_id=task_id,
                event_type=event.get("type"),
                delta_length=len(event.get("text", "")),
                value_count=len(event.get("values", {})) if isinstance(event.get("values"), dict) else 0,
                error=event.get("error", ""),
            )
            yield sse_event(event)
        log_vision("vision_stream.completed", task_id=task_id)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/{task_id}/confirm", response_class=HTMLResponse)
async def confirm_vision_result(request: Request, task_id: str):
    log_vision("vision_confirm.received", task_id=task_id)
    task = pop_vision_task(task_id)
    if not task:
        log_vision("vision_confirm.task_missing", task_id=task_id)
        return RedirectResponse("/", status_code=303)
    model = load_model(task["model_id"])
    if model is None:
        log_vision("vision_confirm.model_missing", task_id=task_id, model_id=task["model_id"])
        return HTMLResponse("<h2>Model not found.</h2>", status_code=404)

    form = await request.form()
    features = form.getlist("feature")
    values = form.getlist("value")
    vision_values = {
        feature: value.strip()
        for feature, value in zip(features, values)
        if feature and value and value.strip()
    }
    notes = str(form.get("notes", ""))
    if task.get("mode") == "update_session":
        update_vision_suggestions(vision_values, notes, "")
    else:
        start_session(model, vision_values, notes, "")
    log_vision(
        "vision_confirm.start_session",
        task_id=task_id,
        model_id=model["id"],
        mode=task.get("mode", "new_session"),
        value_count=len(vision_values),
        values=vision_values,
        notes_length=len(notes),
    )
    return render_next_step(request)
