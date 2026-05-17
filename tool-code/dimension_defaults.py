from __future__ import annotations

from typing import Any

from app_config import BASE_DIR, read_json, write_json


DIMENSION_DEFAULTS_FILE = BASE_DIR / "dimension_defaults.json"

BUILT_IN_DIMENSION_DEFAULTS = {
    "Products ID": {"ease": "N/A", "comments": ""},
    "Name": {"ease": "N/A", "comments": ""},
    "Castrads SKU": {"ease": "N/A", "comments": ""},
    "Family": {
        "ease": "Low",
        "comments": "Sometimes can be found written on the radiator or the radiator bush",
    },
    "Manufacturer": {
        "ease": "Medium",
        "comments": "Sometimes can be found written on the radiator or the radiator bush",
    },
    "Tags": {
        "ease": "N/A",
        "comments": "Some tags may be useful here such as number of columns or ornate",
    },
    "Section Length (mm)": {"ease": "High", "comments": "Easy to measure with a tape measure"},
    "Leg Section Depth (mm)": {"ease": "High", "comments": "Easy to measure with a tape measure"},
    "Mid Section Depth (mm)": {"ease": "High", "comments": "Easy to measure with a tape measure"},
    "Leg Section Height (mm)": {"ease": "High", "comments": "Easy to measure with a tape measure"},
    "Mid Section Height (mm)": {"ease": "High", "comments": "Easy to measure with a tape measure"},
    "Leg Section Weight (kg)": {
        "ease": "Medium",
        "comments": "Might be possible to weigh the whole radiator and divide by the number of sections. Would need to account for split of legs and mids",
    },
    "Mid Section Weight (kg)": {
        "ease": "Medium",
        "comments": "Might be possible to weigh the whole radiator and divide by the number of sections. Would need to account for split of legs and mids",
    },
    "Internal Volume (L)": {
        "ease": "Low",
        "comments": "Need to weigh radiator empty, fill with water, then weight again and divide by number of sections.",
    },
    'Nipple Size Top (")': {
        "ease": "Medium",
        "comments": "May require the user to unscrew a radiator bush. Care needed as 1.25\" means a 1.61\" diameter opening. Perhaps an easier thing to observe here is whether the top and bottom openings are the same size or different.",
    },
    'Nipple Size Bottom (")': {
        "ease": "Medium",
        "comments": "May require the user to unscrew a radiator bush. Care needed as 1.25\" means a 1.61\" diameter opening. Perhaps an easier thing to observe here is whether the top and bottom openings are the same size or different.",
    },
    "Pipe Centre Top To Floor (mm)": {
        "ease": "High",
        "comments": "Possible to measure with tape, but need to make sure are measuring a radiator with legs (not wall mounted)",
    },
    "Pipe Centre Bottom To Floor (mm)": {
        "ease": "High",
        "comments": "Possible to measure with tape, but need to make sure are measuring a radiator with legs (not wall mounted)",
    },
    "Inter Axis Distance (mm)": {"ease": "High", "comments": "Easy to measure with a tape measure"},
    "Exponent N": {"ease": "Very Low", "comments": "Requires laboratory grade equipment"},
    "Factor Km": {"ease": "Very Low", "comments": "Requires laboratory grade equipment"},
    "Output Dt50 (W)": {"ease": "Very Low", "comments": "Requires laboratory grade equipment"},
}


def load_dimension_defaults() -> dict[str, dict[str, str]]:
    saved = read_json(DIMENSION_DEFAULTS_FILE, {})
    merged = {name: config.copy() for name, config in BUILT_IN_DIMENSION_DEFAULTS.items()}
    for name, config in saved.items():
        measurement_comments = config.get("measurement_comments", config.get("comments", ""))
        merged[name] = {
            "ease": config.get("ease", "Medium"),
            "measurement_comments": measurement_comments,
            "image_description": config.get("image_description", measurement_comments),
        }
    for config in merged.values():
        measurement_comments = config.get("measurement_comments", config.get("comments", ""))
        config["measurement_comments"] = measurement_comments
        config["image_description"] = config.get("image_description", measurement_comments)
        config.pop("comments", None)
    return merged


def save_dimension_defaults(defaults: dict[str, dict[str, str]]) -> None:
    write_json(DIMENSION_DEFAULTS_FILE, defaults)


def dimension_default_rows() -> list[dict[str, str]]:
    defaults = load_dimension_defaults()
    return [
        {
            "name": name,
            "ease": config.get("ease", "Medium"),
            "measurement_comments": config.get("measurement_comments", ""),
            "image_description": config.get("image_description", ""),
        }
        for name, config in sorted(defaults.items(), key=lambda item: item[0].lower())
    ]


def dimension_ease_config(column: str, defaults: dict[str, dict[str, str]] | None = None) -> dict[str, str]:
    defaults = defaults or load_dimension_defaults()
    return defaults.get(
        column,
        {"ease": "Medium", "measurement_comments": "", "image_description": ""},
    )


def dimension_default_weight(column: str, defaults: dict[str, dict[str, str]] | None = None) -> float:
    return 2.0 if dimension_ease_config(column, defaults)["ease"].lower() == "high" else 1.0


def dimension_default_enabled(column: str, defaults: dict[str, dict[str, str]] | None = None) -> bool:
    ease = dimension_ease_config(column, defaults)["ease"].lower()
    return ease not in {"n/a", "low", "very low"}


def build_defaults_from_form(
    names: list[str],
    eases: list[str],
    measurement_comments: list[str],
    image_descriptions: list[str],
) -> dict[str, dict[str, str]]:
    allowed = {"N/A", "Low", "Medium", "High", "Very Low"}
    defaults: dict[str, dict[str, str]] = {}
    for index, raw_name in enumerate(names):
        name = raw_name.strip()
        if not name:
            continue
        ease = eases[index] if index < len(eases) else "Medium"
        if ease not in allowed:
            ease = "Medium"
        defaults[name] = {
            "ease": ease,
            "measurement_comments": (
                measurement_comments[index].strip() if index < len(measurement_comments) else ""
            ),
            "image_description": (
                image_descriptions[index].strip() if index < len(image_descriptions) else ""
            ),
        }
    return defaults
