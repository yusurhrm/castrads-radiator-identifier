from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app_config import (
    ACTIVE_MODEL_FILE,
    BASE_DIR,
    DEFAULT_CONFIDENCE_THRESHOLD,
    MODELS_DIR,
    ensure_dirs,
    now_id,
    read_json,
    write_json,
)
from decision_tree import (
    TreeNode,
    build_tree,
    evaluate_tree,
    infer_dimensions,
    is_numeric_dimension,
    load_dataframe,
)
from dimension_defaults import load_dimension_defaults
from vision_service import default_image_understanding, get_image_understanding_config


def train_model(
        source_file: Path,
        dimensions: list[dict[str, Any]],
        name: str,
        threshold: float,
) -> dict[str, Any]:
    ensure_dirs()
    df = load_dataframe(source_file)
    target_col = df.columns[0]
    enabled_dimensions = [d for d in dimensions if d.get("enabled")]
    features = [d["name"] for d in enabled_dimensions if d["name"] in df.columns]
    weights = {d["name"]: float(d.get("weight", 1)) for d in enabled_dimensions}
    feature_types = {
        d["name"]: d.get("type", "categorical") for d in enabled_dimensions
    }

    tree = build_tree(df, features, target_col, weights, feature_types)
    metrics = evaluate_tree(tree, df, target_col)
    model_id = now_id()
    model_dir = MODELS_DIR / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_file, model_dir / "data.xlsx")
    write_json(model_dir / "tree.json", tree.to_dict())
    write_json(model_dir / "metrics.json", metrics)
    write_json(
        model_dir / "config.json",
        {
            "target_column": target_col,
            "dimensions": enabled_dimensions,
            "step_ui": default_step_ui(enabled_dimensions),
            "image_understanding": default_image_understanding(enabled_dimensions),
            "confidence_threshold": threshold,
            "algorithm_version": 2,
        },
    )
    metadata = {
        "id": model_id,
        "name": name or f"Model {model_id}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_filename": source_file.name,
        "rows": len(df),
        "columns": len(df.columns),
        "active": False,
    }
    write_json(model_dir / "metadata.json", metadata)
    return metadata | {"metrics": metrics}


def default_step_ui(dimensions: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    step_ui = {}
    for dimension in dimensions:
        name = dimension["name"]
        step_ui[name] = {
            "display_name": dimension.get("display_name") or name,
            "input_mode": "auto",
            "image_url": default_image_for_feature(name),
            "help_text": "",
        }
    return step_ui


def default_image_for_feature(feature: str) -> str:
    defaults = {
        "Section Length (mm)": "/static/sectionlength.png",
        "Leg Section Depth (mm)": "/static/legsectiondepth.png",
        "Mid Section Height (mm)": "/static/midsectionheight.png",
    }
    return defaults.get(feature, "")


def dimension_lookup(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {d["name"]: d for d in config.get("dimensions", [])}


def get_step_ui(config: dict[str, Any], feature: str) -> dict[str, str]:
    dimensions = dimension_lookup(config)
    dimension = dimensions.get(feature, {})
    defaults = {
        "display_name": dimension.get("display_name") or feature,
        "input_mode": "auto",
        "image_url": default_image_for_feature(feature),
        "help_text": "",
    }
    configured = config.get("step_ui", {}).get(feature, {})
    return defaults | configured


def effective_input_mode(node: TreeNode, configured_mode: str) -> str:
    if configured_mode == "text":
        return "text"
    if configured_mode == "number":
        return "number"
    if configured_mode == "select":
        return "select"
    if node.split_type == "numeric":
        return "number"
    return "select"


def list_models() -> list[dict[str, Any]]:
    ensure_dirs()
    active_id = get_active_model_id()
    models = []
    for model_dir in MODELS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        metadata = read_json(model_dir / "metadata.json", {})
        metrics = read_json(model_dir / "metrics.json", {})
        if metadata:
            metadata["metrics"] = metrics
            metadata["active"] = metadata.get("id") == active_id
            models.append(metadata)
    return sorted(models, key=lambda item: item.get("created_at", ""), reverse=True)


def get_model_metrics(model: dict[str, Any]) -> dict[str, Any]:
    metrics = model.get("metrics", {})
    if "details" in metrics:
        return metrics
    metrics = evaluate_tree(model["tree"], model["data"], model["config"]["target_column"])
    write_json(model["dir"] / "metrics.json", metrics)
    model["metrics"] = metrics
    return metrics


def get_active_model_id() -> str | None:
    data = read_json(ACTIVE_MODEL_FILE, {})
    return data.get("model_id")


def set_active_model(model_id: str) -> None:
    write_json(ACTIVE_MODEL_FILE, {"model_id": model_id})


def clear_active_model() -> None:
    if ACTIVE_MODEL_FILE.exists():
        ACTIVE_MODEL_FILE.unlink()


def load_active_model() -> dict[str, Any] | None:
    model_id = get_active_model_id()
    return load_model(model_id) if model_id else None


def load_model(model_id: str | None) -> dict[str, Any] | None:
    if not model_id:
        return None
    model_dir = MODELS_DIR / model_id
    if not model_dir.exists():
        return None
    config = read_json(model_dir / "config.json", {})
    data = load_dataframe(model_dir / "data.xlsx")
    config = normalize_model_config(config, data)
    return {
        "id": model_id,
        "dir": model_dir,
        "metadata": read_json(model_dir / "metadata.json", {}),
        "config": config,
        "metrics": read_json(model_dir / "metrics.json", {}),
        "tree": TreeNode.from_dict(read_json(model_dir / "tree.json", {})),
        "data": data,
    }


def save_model_config(model_id: str, config: dict[str, Any]) -> None:
    write_json(MODELS_DIR / model_id / "config.json", config)


def delete_model(model_id: str) -> None:
    model_dir = MODELS_DIR / model_id
    if not model_dir.exists():
        return
    shutil.rmtree(model_dir)
    asset_dir = BASE_DIR / "static" / "model_assets" / model_id
    if asset_dir.exists():
        shutil.rmtree(asset_dir)
    if get_active_model_id() == model_id:
        remaining = list_models()
        if remaining:
            set_active_model(remaining[0]["id"])
        else:
            clear_active_model()


def dimensions_for_retrain(model: dict[str, Any]) -> list[dict[str, Any]]:
    inferred = {item["name"]: item for item in infer_dimensions(model["data"], load_dimension_defaults())}
    configured = {item["name"]: item for item in model["config"].get("dimensions", [])}
    dimensions = []
    for name, base in inferred.items():
        dimension = configured.get(name, {})
        dimensions.append(
            {
                "name": name,
                "display_name": dimension.get("display_name") or name,
                "weight": dimension.get("weight", 1),
                "type": dimension.get("type", base.get("type", "categorical")),
                "enabled": dimension.get("enabled", name in configured),
                "ease": dimension.get("ease", base.get("ease", "Medium")),
                "ease_comments": dimension.get("ease_comments", base.get("ease_comments", "")),
                "image_description": dimension.get("image_description", base.get("image_description", "")),
                "valid_count": base.get("valid_count", 0),
                "unique_count": base.get("unique_count", 0),
            }
        )
    return dimensions


def normalize_model_config(config: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    for dimension in config.get("dimensions", []):
        name = dimension.get("name")
        if not name:
            continue
        if not dimension.get("display_name"):
            dimension["display_name"] = name
        if not dimension.get("type"):
            dimension["type"] = "numeric" if name in df.columns and is_numeric_dimension(df[name]) else "categorical"
        if "image_description" not in dimension:
            dimension["image_description"] = dimension.get("ease_comments", "")
    if "step_ui" not in config:
        config["step_ui"] = default_step_ui(config.get("dimensions", []))
    else:
        defaults = default_step_ui(config.get("dimensions", []))
        for name, default_ui in defaults.items():
            if name not in config["step_ui"]:
                config["step_ui"][name] = default_ui
    config["image_understanding"] = get_image_understanding_config(config)
    return config


def bootstrap_default_model() -> None:

    ensure_dirs()

    active_id = get_active_model_id()

    if active_id:

        active_config = read_json(
            MODELS_DIR / active_id / "config.json",
            {}
        )

        if active_config.get("algorithm_version") == 2:
            return

    # UPDATED EXCEL FILE
    excel_file = BASE_DIR / "updated_radiator_data.xlsx"

    if not excel_file.exists():
        return

    if not active_id and list_models():

        set_active_model(list_models()[0]["id"])

        return

    df = load_dataframe(excel_file)

    dimensions = infer_dimensions(df)

    metadata = train_model(
        excel_file,
        dimensions,
        "Updated Radiator Model",
        DEFAULT_CONFIDENCE_THRESHOLD,
    )

    set_active_model(metadata["id"])
