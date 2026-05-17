from __future__ import annotations

from typing import Any

import pandas as pd

from decision_tree import (
    TreeNode,
    choose_split,
    format_threshold,
    is_manufacturer_column,
    majority_prediction,
    normalize_submitted_value,
    sorted_options,
    top_target_predictions,
)
from model_store import effective_input_mode, get_step_ui
from app_config import MISSING

AMBIGUOUS_TOP_K = 5

session_state: dict[str, Any] = {
    "node": None,
    "data": None,
    "history": [],
    "actions": [],
    "asked_features": [],
    "skipped_features": [],
    "model": None,
    "vision_values": {},
    "vision_notes": "",
    "vision_error": "",
}


def get_local_image_url(product_row) -> str:

    image_filename = product_row.get("Image", "")

    if not image_filename or str(image_filename) == MISSING:
        return ""

    return f"/static/radiator_images/{image_filename}"


# =========================================================
# START SESSION
# =========================================================
def start_session(
    model: dict[str, Any],
    vision_values: dict[str, Any] | None = None,
    vision_notes: str = "",
    vision_error: str = "",
) -> None:

    session_state["data"] = model["data"].copy()

    session_state["history"] = []

    session_state["actions"] = []

    session_state["asked_features"] = []

    session_state["skipped_features"] = []

    session_state["model"] = model

    session_state["vision_values"] = vision_values or {}

    session_state["vision_notes"] = vision_notes

    session_state["vision_error"] = vision_error

    session_state["node"] = make_next_node(model)


# =========================================================
# CURRENT CANDIDATE SUMMARY
# =========================================================
def current_candidates_summary(
    df: pd.DataFrame,
    target_col: str
) -> dict[str, Any]:

    label, confidence = majority_prediction(df, target_col)

    product_row = df[df[target_col] == label].iloc[0]

    product_info = {
        col: product_row[col]
        for col in df.columns
        if str(product_row[col]) != MISSING
    }

    image_url = get_local_image_url(product_row)

    return {
        "result": label,
        "confidence": round(confidence, 4),
        "candidate_count": len(df),
        "product_info": product_info,
        "image_url": image_url,
    }


# =========================================================
# RESULT CONTEXT
# =========================================================
def get_result_context(
    summary: dict[str, Any],
    reason: str,
    model: dict[str, Any]
) -> dict[str, Any]:

    return {
        **summary,
        "reason": reason,
        "model": model["metadata"],
        "history": session_state["history"],
        "skipped_features": session_state["skipped_features"],
    }


# =========================================================
# NEXT STEP CONTEXT
# =========================================================
def get_next_step_context(
    model: dict[str, Any],
    confidence_threshold: float,
) -> tuple[str, dict[str, Any]]:

    current_df = session_state.get("data")

    if current_df is None or len(current_df) == 0:
        return "no_match", {}

    node = session_state.get("node")

    if node is None:
        return "no_node", {}

    target_col = model["config"]["target_column"]

    summary = current_candidates_summary(
        current_df,
        target_col
    )

    # =====================================================
    # QUESTION STEP
    # =====================================================
    if node.feature:

        if summary["confidence"] >= confidence_threshold:

            return "result", get_result_context(
                summary,
                "Confidence threshold reached",
                model,
            )

        feature = node.feature

        step_ui = get_step_ui(
            model["config"],
            feature
        )

        input_mode = effective_input_mode(
            node,
            step_ui.get("input_mode", "auto"),
        )

        progress = progress_context(model)

        return "question", {
            "feature": feature,
            "display_feature": step_ui["display_name"],
            "options": sorted_options(current_df[feature])
            if node.split_type != "numeric"
            else [],
            "input_mode": input_mode,
            "split_type": node.split_type,
            "node_threshold": format_threshold(node.threshold),
            "guide_image_url": step_ui.get("image_url", ""),
            "help_text": step_ui.get("help_text", ""),
            "default_value": default_vision_value(feature),
            "vision_suggestions": vision_suggestion_rows(feature),
            "vision_notes": session_state.get("vision_notes", ""),
            "vision_error": session_state.get("vision_error", ""),
            "confidence": summary["confidence"],
            "candidate_count": summary["candidate_count"],
            "threshold": confidence_threshold,
            "model": model["metadata"],
            "history": session_state["history"],
            **progress,
        }

    # =====================================================
    # SINGLE RESULT
    # =====================================================
    if summary["confidence"] >= confidence_threshold:

        reason = (
            "Decision tree reached a leaf"
            if node.label
            else "All dimensions completed"
        )

        return "result", get_result_context(
            summary,
            reason,
            model,
        )

    # =====================================================
    # AMBIGUOUS RESULTS
    # =====================================================
    top_candidates = top_target_predictions(
        current_df,
        target_col,
        k=AMBIGUOUS_TOP_K,
    )

    enhanced_candidates = []

    for c in top_candidates:

        row = current_df[
            current_df[target_col] == c["label"]
        ].iloc[0]

        image_url = get_local_image_url(row)

        product_info = {
            col: row[col]
            for col in current_df.columns
            if str(row[col]) != MISSING
        }

        enhanced_candidates.append({
            **c,
            "image_url": image_url,
            "product_info": product_info,
        })

    top_candidates = enhanced_candidates

    ctx = get_result_context(
        summary,
        "Confidence below threshold — compare candidates below",
        model,
    )

    ctx["ambiguous"] = True

    ctx["top_candidates"] = top_candidates

    ctx["threshold"] = confidence_threshold

    return "ambiguous", ctx


# =========================================================
# APPLY ANSWER
# =========================================================
def apply_answer(
    feature: str,
    value: str,
    record: bool = True
) -> None:

    current_df = session_state.get("data")

    node = session_state.get("node")

    if current_df is None or node is None or node.feature != feature:
        return

    model = session_state.get("model")

    if (
        node.split_type != "numeric"
        and model is not None
        and is_manufacturer_column(feature)
        and value.strip().lower() in {"", "unknown"}
    ):
        skip_feature(feature, model)
        return

    if record:
        session_state["actions"].append({
            "feature": feature,
            "value": value
        })

    session_state["asked_features"].append(feature)

    # =====================================================
    # NUMERIC SPLIT
    # =====================================================
    if node.split_type == "numeric":

        submitted_number = pd.to_numeric(
            pd.Series([value]),
            errors="coerce"
        ).iloc[0]

        if pd.isna(submitted_number) or node.threshold is None:

            session_state["history"].append((feature, value))

            session_state["data"] = current_df.iloc[0:0]

            return

        branch = (
            "le"
            if float(submitted_number) <= float(node.threshold)
            else "gt"
        )

        operator = "<=" if branch == "le" else ">"

        session_state["history"].append(
            (
                feature,
                f"{value} ({operator} {format_threshold(node.threshold)})"
            )
        )

        numeric = pd.to_numeric(
            current_df[feature],
            errors="coerce"
        )

        mask = (
            numeric <= node.threshold
            if branch == "le"
            else numeric > node.threshold
        )

        session_state["data"] = current_df[mask]

        session_state["node"] = make_next_node_from_current(
            model_context(),
            session_state["data"],
        )

        return

    # =====================================================
    # CATEGORICAL SPLIT
    # =====================================================
    matched_value = normalize_submitted_value(
        current_df[feature],
        value,
    )

    session_state["history"].append(
        (feature, matched_value or value)
    )

    if matched_value is None:

        session_state["data"] = current_df.iloc[0:0]

    else:

        session_state["data"] = current_df[
            current_df[feature] == matched_value
        ]

        session_state["node"] = make_next_node_from_current(
            model_context(),
            session_state["data"],
        )


# =========================================================
# SKIP FEATURE
# =========================================================
def skip_feature(
    feature: str,
    model: dict[str, Any]
) -> None:

    current_df = session_state.get("data")

    if current_df is None:
        return

    session_state["asked_features"].append(feature)

    session_state["history"].append(
        (feature, "SKIPPED")
    )

    session_state["actions"].append({
        "feature": feature,
        "value": "SKIPPED"
    })

    session_state["skipped_features"].append(feature)

    session_state["node"] = make_next_sequential_node(
        model,
        current_df,
    )


# =========================================================
# GO BACK
# =========================================================
def go_back(model: dict[str, Any]) -> bool:

    if not session_state["history"]:
        return False

    session_state["actions"].pop()

    replay = list(session_state["actions"])

    vision_values = dict(
        session_state.get("vision_values", {})
    )

    vision_notes = session_state.get("vision_notes", "")

    vision_error = session_state.get("vision_error", "")

    start_session(
        model,
        vision_values,
        vision_notes,
        vision_error,
    )

    for action in replay:

        feature = action["feature"]

        value = action["value"]

        if value == "SKIPPED":

            skip_feature(feature, model)

            session_state["actions"].pop()

        else:

            apply_answer(
                feature,
                value,
                record=False
            )

            session_state["actions"].append(action)

    return True


# =========================================================
# CAN ANSWER
# =========================================================
def can_answer(feature: str) -> bool:

    node = session_state.get("node")

    return (
        node is not None
        and node.feature == feature
    )


# =========================================================
# UPDATE VISION SUGGESTIONS
# =========================================================
def update_vision_suggestions(
    values: dict[str, Any],
    notes: str = "",
    error: str = "",
) -> None:

    existing = dict(
        session_state.get("vision_values", {})
    )

    for feature, value in values.items():

        if value not in (None, ""):
            existing[feature] = value

    session_state["vision_values"] = existing

    if notes:
        session_state["vision_notes"] = notes

    if error:
        session_state["vision_error"] = error


# =========================================================
# DEFAULT VISION VALUE
# =========================================================
def default_vision_value(feature: str) -> Any:

    if feature in session_state.get("asked_features", []):
        return ""

    return session_state.get(
        "vision_values",
        {}
    ).get(feature, "")


# =========================================================
# VISION SUGGESTION ROWS
# =========================================================
def vision_suggestion_rows(
    current_feature: str | None = None
) -> list[dict[str, Any]]:

    suggestions = []

    asked = set(
        session_state.get("asked_features", [])
    )

    for feature, value in session_state.get(
        "vision_values",
        {}
    ).items():

        suggestions.append({
            "feature": feature,
            "value": value,
            "completed": feature in asked,
            "current": feature == current_feature,
        })

    return suggestions


# =========================================================
# PROGRESS CONTEXT
# =========================================================
def progress_context(
    model: dict[str, Any]
) -> dict[str, Any]:

    total_steps = max(
        1,
        sum(
            1
            for d in model["config"].get("dimensions", [])
            if d.get("enabled", True)
        ),
    )

    completed_steps = min(
        len(session_state["asked_features"]),
        total_steps,
    )

    current_step = min(
        completed_steps + 1,
        total_steps
    )

    progress_percent = round(
        (completed_steps / total_steps) * 100
    )

    return {
        "current_step": current_step,
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "progress_percent": progress_percent,
    }


# =========================================================
# MAKE NEXT NODE
# =========================================================
def make_next_node(
    model: dict[str, Any]
) -> TreeNode:

    return make_next_node_from_current(
        model,
        session_state["data"],
    )


# =========================================================
# MAKE NEXT NODE FROM CURRENT
# =========================================================
def make_next_node_from_current(
    model: dict[str, Any] | None,
    current_df: pd.DataFrame | None,
) -> TreeNode | None:

    if model is None or current_df is None or len(current_df) == 0:
        return None

    target_col = model["config"]["target_column"]

    label, confidence = majority_prediction(
        current_df,
        target_col,
    )

    if len(set(current_df[target_col])) == 1:

        return TreeNode(
            label=label,
            confidence=confidence,
            sample_count=len(current_df),
        )

    dimensions = [
        d
        for d in model["config"].get("dimensions", [])
        if d.get("enabled", True)
        and d["name"] not in session_state["asked_features"]
    ]

    features = [
        d["name"]
        for d in dimensions
        if d["name"] in current_df.columns
    ]

    if not features:

        return TreeNode(
            label=label,
            confidence=confidence,
            sample_count=len(current_df),
        )

    weights = {
        d["name"]: float(d.get("weight", 1))
        for d in dimensions
    }

    feature_types = {
        d["name"]: d.get("type", "categorical")
        for d in dimensions
    }

    split = choose_split(
        current_df,
        features,
        target_col,
        weights,
        feature_types,
    )

    if split is None:

        return TreeNode(
            label=label,
            confidence=confidence,
            sample_count=len(current_df),
        )

    return TreeNode(
        feature=split["feature"],
        confidence=confidence,
        sample_count=len(current_df),
        split_type=split["split_type"],
        threshold=split["threshold"],
    )


# =========================================================
# MAKE NEXT SEQUENTIAL NODE
# =========================================================
def make_next_sequential_node(
    model: dict[str, Any] | None,
    current_df: pd.DataFrame | None,
) -> TreeNode | None:

    if model is None or current_df is None or len(current_df) == 0:
        return None

    target_col = model["config"]["target_column"]

    label, confidence = majority_prediction(
        current_df,
        target_col,
    )

    if len(set(current_df[target_col])) == 1:

        return TreeNode(
            label=label,
            confidence=confidence,
            sample_count=len(current_df),
        )

    for dimension in model["config"].get("dimensions", []):

        name = dimension["name"]

        if not dimension.get("enabled", True):
            continue

        if (
            name in session_state["asked_features"]
            or name not in current_df.columns
        ):
            continue

        if current_df[name].nunique() <= 1:
            continue

        feature_type = dimension.get(
            "type",
            "categorical"
        )

        if feature_type == "numeric":

            _, threshold = numeric_threshold_for_sequential_node(
                current_df,
                name,
                target_col,
            )

            if threshold is None:
                continue

            return TreeNode(
                feature=name,
                confidence=confidence,
                sample_count=len(current_df),
                split_type="numeric",
                threshold=threshold,
            )

        return TreeNode(
            feature=name,
            confidence=confidence,
            sample_count=len(current_df),
            split_type="categorical",
            threshold=None,
        )

    return TreeNode(
        label=label,
        confidence=confidence,
        sample_count=len(current_df),
    )


# =========================================================
# NUMERIC THRESHOLD
# =========================================================
def numeric_threshold_for_sequential_node(
    current_df: pd.DataFrame,
    feature: str,
    target_col: str,
) -> tuple[float, float | None]:

    current_node = session_state.get("node")

    if (
        current_node
        and current_node.feature == feature
        and current_node.threshold is not None
    ):
        return 0.0, current_node.threshold

    numeric = pd.to_numeric(
        current_df[feature],
        errors="coerce",
    ).dropna()

    unique_values = sorted(numeric.unique())

    if len(unique_values) <= 1:
        return 0.0, None

    middle = len(unique_values) // 2

    left = unique_values[middle - 1]

    right = unique_values[middle]

    return 0.0, (
        float(left) + float(right)
    ) / 2


# =========================================================
# MODEL CONTEXT
# =========================================================
def model_context() -> dict[str, Any] | None:
    return session_state.get("model")