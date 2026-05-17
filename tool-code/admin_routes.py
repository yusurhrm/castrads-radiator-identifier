from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from app_config import BASE_DIR, DEFAULT_CONFIDENCE_THRESHOLD, UPLOADS_DIR, ensure_dirs, now_id
from decision_tree import collect_tree_steps, infer_dimensions, load_dataframe
from dimension_defaults import (
    build_defaults_from_form,
    dimension_default_rows,
    load_dimension_defaults,
    save_dimension_defaults,
)
from model_store import (
    delete_model,
    dimensions_for_retrain,
    get_active_model_id,
    get_model_metrics,
    get_step_ui,
    list_models,
    load_model,
    save_model_config,
    set_active_model,
    train_model,
)
from vision_service import build_image_prompt, get_image_understanding_config
from web_templates import template_engine


router = APIRouter(prefix="/admin")


@router.get("", response_class=HTMLResponse)
def admin(request: Request):
    return template_engine.TemplateResponse(
        request=request,
        name="admin.html",
        context={"models": list_models(), "active_model_id": get_active_model_id()},
    )


@router.get("/dimension-defaults", response_class=HTMLResponse)
def dimension_defaults_page(request: Request):
    return template_engine.TemplateResponse(
        request=request,
        name="dimension_defaults.html",
        context={"rows": dimension_default_rows()},
    )


@router.post("/dimension-defaults", response_class=HTMLResponse)
async def save_dimension_defaults_page(request: Request):
    form = await request.form()
    defaults = build_defaults_from_form(
        form.getlist("dimension_name"),
        form.getlist("dimension_ease"),
        form.getlist("dimension_measurement_comments"),
        form.getlist("dimension_image_description"),
    )
    save_dimension_defaults(defaults)
    return template_engine.TemplateResponse(
        request=request,
        name="dimension_defaults.html",
        context={"rows": dimension_default_rows(), "message": "Dimension defaults saved."},
    )


@router.post("/upload", response_class=HTMLResponse)
def upload_dataset(request: Request, file: UploadFile = File(...)):
    ensure_dirs()
    upload_id = now_id()
    safe_name = Path(file.filename or "dataset.xlsx").name
    upload_path = UPLOADS_DIR / f"{upload_id}_{safe_name}"
    with upload_path.open("wb") as output:
        shutil.copyfileobj(file.file, output)

    df = load_dataframe(upload_path)
    defaults = load_dimension_defaults()
    return template_engine.TemplateResponse(
        request=request,
        name="train_config.html",
        context={
            "upload_path": str(upload_path),
            "filename": safe_name,
            "rows": len(df),
            "columns": len(df.columns),
            "target_column": df.columns[0],
            "dimensions": infer_dimensions(df, defaults),
            "default_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        },
    )


@router.post("/train", response_class=HTMLResponse)
def train(
    request: Request,
    upload_path: str = Form(...),
    model_name: str = Form(""),
    confidence_threshold: float = Form(DEFAULT_CONFIDENCE_THRESHOLD),
    dimension_name: list[str] = Form(...),
    dimension_display_name: list[str] = Form(...),
    dimension_weight: list[float] = Form(...),
    dimension_type: list[str] = Form(...),
    dimension_ease: list[str] = Form(default=[]),
    dimension_ease_comments: list[str] = Form(default=[]),
    dimension_image_description: list[str] = Form(default=[]),
    enabled_dimension: list[str] = Form(default=[]),
):
    enabled = set(enabled_dimension)
    dimensions = []
    for name, display_name, weight, dim_type in zip(
        dimension_name, dimension_display_name, dimension_weight, dimension_type
    ):
        index = len(dimensions)
        dimensions.append(
            {
                "name": name,
                "display_name": display_name.strip() or name,
                "weight": float(weight),
                "type": dim_type if dim_type in {"numeric", "categorical"} else "categorical",
                "enabled": name in enabled,
                "ease": dimension_ease[index] if index < len(dimension_ease) else "Medium",
                "ease_comments": dimension_ease_comments[index] if index < len(dimension_ease_comments) else "",
                "image_description": (
                    dimension_image_description[index] if index < len(dimension_image_description) else ""
                ),
            }
        )

    metadata = train_model(Path(upload_path), dimensions, model_name, confidence_threshold)
    if not get_active_model_id():
        set_active_model(metadata["id"])
    return template_engine.TemplateResponse(
        request=request,
        name="admin.html",
        context={
            "models": list_models(),
            "active_model_id": get_active_model_id(),
            "message": f"Model {metadata['name']} trained successfully.",
        },
    )


@router.post("/activate")
def activate_model(model_id: str = Form(...)):
    set_active_model(model_id)
    return RedirectResponse("/admin", status_code=303)


@router.get("/model/{model_id}", response_class=HTMLResponse)
def model_detail(request: Request, model_id: str):
    model = load_model(model_id)
    if model is None:
        return HTMLResponse("<h2>Model not found.</h2>", status_code=404)
    metrics = get_model_metrics(model)
    return template_engine.TemplateResponse(
        request=request,
        name="model_detail.html",
        context={
            "model": model["metadata"],
            "config": model["config"],
            "metrics": metrics,
            "active_model_id": get_active_model_id(),
        },
    )


@router.get("/model/{model_id}/retrain", response_class=HTMLResponse)
def retrain_config(request: Request, model_id: str):
    model = load_model(model_id)
    if model is None:
        return HTMLResponse("<h2>Model not found.</h2>", status_code=404)
    return template_engine.TemplateResponse(
        request=request,
        name="train_config.html",
        context={
            "upload_path": str(model["dir"] / "data.xlsx"),
            "filename": model["metadata"].get("source_filename", "data.xlsx"),
            "rows": len(model["data"]),
            "columns": len(model["data"].columns),
            "target_column": model["config"]["target_column"],
            "dimensions": dimensions_for_retrain(model),
            "default_threshold": model["config"].get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD),
            "default_model_name": f"{model['metadata'].get('name', model_id)} retrained",
        },
    )


@router.post("/model/{model_id}/delete")
def remove_model(model_id: str):
    delete_model(model_id)
    return RedirectResponse("/admin", status_code=303)


@router.get("/model/{model_id}/flow", response_class=HTMLResponse)
def configure_flow(request: Request, model_id: str):
    model = load_model(model_id)
    if model is None:
        return HTMLResponse("<h2>Model not found.</h2>", status_code=404)
    return template_engine.TemplateResponse(
        request=request,
        name="flow_config.html",
        context={
            "model": model["metadata"],
            "steps": configured_flow_steps(model),
            "available_images": list_model_images(model_id),
            "vision_prompt": generated_vision_prompt(model),
        },
    )


@router.post("/model/{model_id}/flow", response_class=HTMLResponse)
async def save_flow_config(request: Request, model_id: str):
    model = load_model(model_id)
    if model is None:
        return HTMLResponse("<h2>Model not found.</h2>", status_code=404)

    form = await request.form()
    feature = form.getlist("feature")
    display_name = form.getlist("display_name")
    input_mode = form.getlist("input_mode")
    existing_image_url = form.getlist("existing_image_url")
    selected_image_url = form.getlist("selected_image_url")
    clear_image = set(form.getlist("clear_image"))
    help_text = form.getlist("help_text")
    vision_enabled = set(form.getlist("vision_enabled"))

    config = model["config"]
    step_ui = config.get("step_ui", {})
    image_understanding = get_image_understanding_config(config)
    vision_dimensions = image_understanding.get("dimensions", {})
    for index, name in enumerate(feature):
        mode = input_mode[index] if index < len(input_mode) else "auto"
        if mode not in {"auto", "number", "text", "select"}:
            mode = "auto"
        image_url = existing_image_url[index].strip() if index < len(existing_image_url) else ""
        selected_url = selected_image_url[index].strip() if index < len(selected_image_url) else ""
        if selected_url:
            image_url = selected_url
        upload = form.get(f"image_file_{index}")
        if upload is not None and getattr(upload, "filename", ""):
            image_url = save_step_image(model_id, upload)
        if name in clear_image:
            image_url = ""
        step_ui[name] = {
            "display_name": display_name[index].strip() if index < len(display_name) else name,
            "input_mode": mode,
            "image_url": image_url,
            "help_text": help_text[index].strip() if index < len(help_text) else "",
        }
        vision_dimensions[name] = {
            "enabled": name in vision_enabled,
        }
    config["step_ui"] = step_ui
    image_understanding["dimensions"] = vision_dimensions
    config["image_understanding"] = image_understanding
    save_model_config(model_id, config)

    refreshed = load_model(model_id)
    return template_engine.TemplateResponse(
        request=request,
        name="flow_config.html",
        context={
            "model": refreshed["metadata"],
            "steps": configured_flow_steps(refreshed),
            "available_images": list_model_images(model_id),
            "vision_prompt": generated_vision_prompt(refreshed),
            "message": "Flow UI configuration saved.",
        },
    )


@router.post("/model/{model_id}/asset/upload")
def upload_model_asset(model_id: str, image: UploadFile = File(...)):
    image_url = save_step_image(model_id, image)
    return JSONResponse({"image_url": image_url, "images": list_model_images(model_id)})


@router.post("/model/{model_id}/asset/delete")
def delete_model_asset(model_id: str, image_url: str = Form(...)):
    asset_path = image_url_to_asset_path(model_id, image_url)
    if asset_path and asset_path.exists():
        asset_path.unlink()
    return JSONResponse({"ok": True, "images": list_model_images(model_id)})


def save_step_image(model_id: str, upload: UploadFile) -> str:
    safe_name = Path(upload.filename or "image").name
    asset_dir = BASE_DIR / "static" / "model_assets" / model_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    target = asset_dir / f"{now_id()}_{safe_name}"
    with target.open("wb") as output:
        shutil.copyfileobj(upload.file, output)
    return f"/static/model_assets/{model_id}/{target.name}"


def image_url_to_asset_path(model_id: str, image_url: str) -> Path | None:
    prefix = f"/static/model_assets/{model_id}/"
    if not image_url.startswith(prefix):
        return None
    asset_dir = (BASE_DIR / "static" / "model_assets" / model_id).resolve()
    candidate = (asset_dir / Path(image_url).name).resolve()
    if asset_dir not in candidate.parents:
        return None
    return candidate


def list_model_images(model_id: str) -> list[str]:
    asset_dir = BASE_DIR / "static" / "model_assets" / model_id
    if not asset_dir.exists():
        return []
    allowed = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
    images = [
        f"/static/model_assets/{model_id}/{path.name}"
        for path in asset_dir.iterdir()
        if path.is_file() and path.suffix.lower() in allowed
    ]
    return sorted(images)


def configured_flow_steps(model: dict[str, Any]) -> list[dict[str, Any]]:
    image_config = get_image_understanding_config(model["config"])
    steps = []
    for step in collect_tree_steps(model["tree"]):
        ui = get_step_ui(model["config"], step["feature"])
        vision = image_config.get("dimensions", {}).get(step["feature"], {})
        steps.append(
            step
            | ui
            | {
                "vision_enabled": bool(vision.get("enabled", False)),
            }
        )
    return steps


def generated_vision_prompt(model: dict[str, Any]) -> str:
    prompt, dimensions = build_image_prompt(model)
    if not dimensions:
        return f"{prompt}\n\nNo image-understanding fields are enabled yet."
    return prompt
