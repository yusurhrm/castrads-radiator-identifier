from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

from app_logging import elapsed_ms, log_vision, timed_vision
from app_config import QWEN_API_BASE_URL, QWEN_API_KEY, QWEN_VISION_MODEL, read_json, write_json
from decision_tree import collect_tree_steps

DEFAULT_VISION_MODEL = QWEN_VISION_MODEL
GENERIC_IMAGE_PROMPT_TEMPLATE = """You analyze one or more photos of the same radiator/product.
Extract only the configured fields listed below.
Use the field display name, input mode, help text, decision threshold, and allowed values to understand what to look for.
Return unknown or not-visible values as null. Do not guess.
For numeric fields, return a plain number as a string without units.
For categorical fields, prefer one of the allowed values exactly when provided.
Return only valid JSON in this shape: {"values": {"Field Name": "value or null"}, "notes": "short note"}."""


def default_image_understanding(dimensions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": DEFAULT_VISION_MODEL,
        "dimensions": {
            dimension["name"]: {
                "enabled": False,
            }
            for dimension in dimensions
        },
    }


def get_image_understanding_config(config: dict[str, Any]) -> dict[str, Any]:
    defaults = default_image_understanding(config.get("dimensions", []))
    configured = config.get("image_understanding", {})
    configured_model = configured.get("model") or ""
    if configured_model.startswith("gpt-"):
        configured_model = ""
    merged_dimensions = defaults["dimensions"]
    for name, value in configured.get("dimensions", {}).items():
        if name in merged_dimensions:
            merged_dimensions[name] = merged_dimensions[name] | value
    return {
        "model": configured_model or defaults["model"],
        "dimensions": merged_dimensions,
    }


def image_understanding_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    image_config = get_image_understanding_config(config)
    step_ui = config.get("step_ui", {})
    rows = []
    for dimension in config.get("dimensions", []):
        name = dimension["name"]
        ui = step_ui.get(name, {})
        item = image_config["dimensions"].get(name, {})
        rows.append(
            {
                "name": name,
                "display_name": ui.get("display_name") or dimension.get("display_name") or name,
                "type": dimension.get("type", "categorical"),
                "enabled": bool(item.get("enabled", False)),
            }
        )
    return rows


def save_image_understanding_config(model_dir: Path, config: dict[str, Any]) -> None:
    write_json(model_dir / "config.json", config)


def build_image_prompt(model: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    image_config = get_image_understanding_config(model["config"])
    enabled_dimensions = []
    dimension_map = {d["name"]: d for d in model["config"].get("dimensions", [])}
    step_ui = model["config"].get("step_ui", {})
    step_map = {step["feature"]: step for step in collect_tree_steps(model["tree"])}

    for name, item in image_config.get("dimensions", {}).items():
        if not item.get("enabled") or name not in dimension_map:
            continue
        dimension = dimension_map[name]
        ui = step_ui.get(name, {})
        step = step_map.get(name, {})
        values = []
        if dimension.get("type") != "numeric" and name in model["data"].columns:
            values = sorted(
                str(value)
                for value in model["data"][name].dropna().drop_duplicates().tolist()
                if str(value).strip()
            )[:80]
        enabled_dimensions.append(
            {
                "name": name,
                "display_name": ui.get("display_name")
                or dimension.get("display_name")
                or name,
                "type": dimension.get("type", "categorical"),
                "description": dimension.get("image_description", ""),
                "input_mode": ui.get("input_mode", "auto"),
                "help_text": ui.get("help_text", ""),
                "split_type": step.get("split_type", dimension.get("type", "categorical")),
                "threshold": step.get("threshold", ""),
                "allowed_values": values,
            }
        )

    lines = [GENERIC_IMAGE_PROMPT_TEMPLATE, "", "Configured fields:"]
    for dimension in enabled_dimensions:
        details = [
            f"field_key={dimension['name']}",
            f"display_name={dimension['display_name']}",
            f"type={dimension['type']}",
            f"input_mode={dimension['input_mode']}",
        ]
        if dimension["description"]:
            details.append(f"field_description={dimension['description']}")
        if dimension["split_type"]:
            details.append(f"decision_split={dimension['split_type']}")
        if dimension["threshold"]:
            details.append(f"decision_threshold={dimension['threshold']}")
        if dimension["help_text"]:
            details.append(f"help_text={dimension['help_text']}")
        if dimension["allowed_values"]:
            details.append(f"allowed_values={', '.join(dimension['allowed_values'])}")
        lines.append(f"- {'; '.join(details)}")
    return "\n".join(lines), enabled_dimensions


def image_to_data_url(image_path: Path) -> str:
    with timed_vision("vision_service.image_to_data_url", image_path=str(image_path)):
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        raw = image_path.read_bytes()
        encoded = base64.b64encode(raw).decode("ascii")
    log_vision(
        "vision_service.image_encoded",
        image_path=str(image_path),
        mime_type=mime_type,
        file_bytes=len(raw),
        base64_chars=len(encoded),
    )
    return f"data:{mime_type};base64,{encoded}"


def analyze_images(model: dict[str, Any], image_paths: list[Path]) -> dict[str, Any]:
    api_key = QWEN_API_KEY.strip() or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if not api_key:
        return {
            "ok": False,
            "error": "QWEN_API_KEY is not configured.",
            "values": {},
            "notes": "",
        }
    if not image_paths:
        return {"ok": True, "error": "", "values": {}, "notes": ""}

    prompt, dimensions = build_image_prompt(model)
    if not dimensions:
        return {
            "ok": False,
            "error": "No image-understanding dimensions are enabled for this model.",
            "values": {},
            "notes": "",
        }

    image_config = get_image_understanding_config(model["config"])
    content = [
        {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}}
    for image_path in image_paths
    ]
    content.append({"type": "text", "text": prompt})

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=QWEN_API_BASE_URL.rstrip("/"),
            timeout=60,
        )
        completion = client.chat.completions.create(
            model=image_config.get("model") or DEFAULT_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": content,
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Qwen image understanding failed: {exc}",
            "values": {},
            "notes": "",
        }

    content_value = completion.choices[0].message.content if completion.choices else ""
    if isinstance(content_value, str):
        text = content_value
    elif isinstance(content_value, list):
        text = "".join(
            part.get("text", "") for part in content_value if isinstance(part, dict)
        )
    else:
        text = ""
    if not text:
        return {"ok": False, "error": "Qwen response did not contain message content.", "values": {}, "notes": ""}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"ok": False, "error": "Qwen response was not valid JSON.", "values": {}, "notes": text}

    values = {
        key: value
        for key, value in parsed.get("values", {}).items()
        if value not in (None, "")
    }
    return {
        "ok": True,
        "error": "",
        "values": values,
        "notes": parsed.get("notes", ""),
        "raw": parsed,
    }


def parse_vision_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"ok": False, "error": "Qwen response was not valid JSON.", "values": {}, "notes": text}
    values = {
        key: value
        for key, value in parsed.get("values", {}).items()
        if value not in (None, "")
    }
    return {
        "ok": True,
        "error": "",
        "values": values,
        "notes": parsed.get("notes", ""),
        "raw": parsed,
    }


def analyze_images_stream(model: dict[str, Any], image_paths: list[Path]):
    total_start = time.perf_counter()
    log_vision(
        "vision_stream_model.enter",
        model_id=model.get("id"),
        image_count=len(image_paths),
        image_paths=[str(path) for path in image_paths],
    )
    api_key = QWEN_API_KEY.strip() or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if not api_key:
        log_vision("vision_stream_model.no_api_key")
        yield {"type": "error", "error": "QWEN_API_KEY is not configured."}
        return
    if not image_paths:
        log_vision("vision_stream_model.no_images")
        yield {"type": "result", "values": {}, "notes": "", "fields": []}
        return

    with timed_vision("vision_stream_model.build_prompt", model_id=model.get("id")):
        prompt, dimensions = build_image_prompt(model)
    log_vision(
        "vision_stream_model.prompt_ready",
        prompt_chars=len(prompt),
        enabled_dimensions=[dimension["name"] for dimension in dimensions],
    )
    if not dimensions:
        log_vision("vision_stream_model.no_dimensions")
        yield {"type": "error", "error": "No image-understanding dimensions are enabled for this model."}
        return

    image_config = get_image_understanding_config(model["config"])
    with timed_vision("vision_stream_model.encode_images", image_count=len(image_paths)):
        content = [
            {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}}
            for image_path in image_paths
        ]
    content.append({"type": "text", "text": prompt})
    log_vision(
        "vision_stream_model.request_ready",
        provider="qwen",
        model=image_config.get("model") or DEFAULT_VISION_MODEL,
        base_url=QWEN_API_BASE_URL.rstrip("/"),
        content_parts=len(content),
    )
    yield {"type": "status", "message": "Images prepared. Calling Qwen image understanding..."}

    try:
        from openai import OpenAI

        with timed_vision("vision_stream_model.client_create", base_url=QWEN_API_BASE_URL.rstrip("/")):
            client = OpenAI(
                api_key=api_key,
                base_url=QWEN_API_BASE_URL.rstrip("/"),
                timeout=60,
            )
        request_start = time.perf_counter()
        log_vision("vision_stream_model.request_start")
        with timed_vision("vision_stream_model.request_create"):
            stream = client.chat.completions.create(
                model=image_config.get("model") or DEFAULT_VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
                stream=True,
            )
        text = ""
        chunk_count = 0
        first_delta_logged = False
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if delta:
                chunk_count += 1
                if not first_delta_logged:
                    first_delta_logged = True
                    log_vision(
                        "vision_stream_model.first_delta",
                        elapsed_ms=elapsed_ms(request_start),
                        delta_chars=len(delta),
                    )
                text += delta
                log_vision("vision_stream_model.delta", chunk_count=chunk_count, delta_chars=len(delta), total_chars=len(text))
                yield {"type": "delta", "text": delta}
        log_vision(
            "vision_stream_model.stream_end",
            elapsed_ms=elapsed_ms(request_start),
            chunk_count=chunk_count,
            total_chars=len(text),
        )
    except Exception as exc:
        log_vision("vision_stream_model.exception", elapsed_ms=elapsed_ms(total_start), error=str(exc))
        yield {"type": "error", "error": f"Qwen image understanding failed: {exc}"}
        return

    with timed_vision("vision_stream_model.parse_json", total_chars=len(text)):
        result = parse_vision_json(text)
    result["type"] = "result" if result.get("ok") else "error"
    result["fields"] = [
        {"name": dimension["name"], "display_name": dimension["display_name"]}
        for dimension in dimensions
    ]
    log_vision(
        "vision_stream_model.result_ready",
        elapsed_ms=elapsed_ms(total_start),
        ok=result.get("ok"),
        value_count=len(result.get("values", {})),
        error=result.get("error", ""),
    )
    yield result


def analyze_image(model: dict[str, Any], image_path: Path) -> dict[str, Any]:
    return analyze_images(model, [image_path])


def read_vision_result(path: Path) -> dict[str, Any]:
    return read_json(path, {})
