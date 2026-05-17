from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app_config import MISSING
from dimension_defaults import (
    dimension_default_enabled,
    dimension_default_weight,
    dimension_ease_config,
    load_dimension_defaults,
)


@dataclass
class TreeNode:
    feature: str | None = None
    label: str | None = None
    confidence: float = 0.0
    sample_count: int = 0
    split_type: str | None = None
    threshold: float | None = None
    children: dict[str, "TreeNode"] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "label": self.label,
            "confidence": self.confidence,
            "sample_count": self.sample_count,
            "split_type": self.split_type,
            "threshold": self.threshold,
            "children": {k: v.to_dict() for k, v in self.children.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TreeNode":
        node = cls(
            feature=data.get("feature"),
            label=data.get("label"),
            confidence=float(data.get("confidence", 0)),
            sample_count=int(data.get("sample_count", 0)),
            split_type=data.get("split_type"),
            threshold=data.get("threshold"),
        )
        node.children = {
            str(k): cls.from_dict(v) for k, v in data.get("children", {}).items()
        }
        return node


def is_manufacturer_column(name: str) -> bool:
    return str(name).strip().lower() in {"manufacturer", "manufacture"}


def _treat_manufacturer_unknown_as_missing(df: pd.DataFrame) -> None:
    """Manufacturer / manufacture: unknown and blanks match empty (same as skipping the dimension)."""
    for col in df.columns:
        if not is_manufacturer_column(col):
            continue
        series = df[col].astype(str)
        stripped = series.str.strip()
        reset = stripped.str.lower().eq("unknown") | stripped.eq("")
        if reset.any():
            df.loc[reset, col] = MISSING


def load_dataframe(path) -> pd.DataFrame:
    df = pd.read_excel(path, dtype=str)
    df = df.fillna(MISSING)
    _treat_manufacturer_unknown_as_missing(df)
    if {"Plain", "Flat top", "Scroll", "Round top"}.issubset(df.columns):
        df["Top Style"] = df.apply(get_top_style, axis=1)
    return df


def get_top_style(row: pd.Series) -> str:
    for column in ["Plain", "Flat top", "Scroll", "Round top"]:
        if str(row.get(column, "")) == "1":
            return column
    return MISSING


def infer_dimensions(df: pd.DataFrame, defaults: dict[str, dict[str, str]] | None = None) -> list[dict[str, Any]]:
    defaults = defaults or load_dimension_defaults()
    target_col = df.columns[0]
    excluded_names = {target_col}
    dimensions = []
    for column in df.columns:
        if column in excluded_names:
            continue
        valid_count = int((df[column] != MISSING).sum())
        unique_count = int(df[column].nunique(dropna=True))
        ease_config = dimension_ease_config(column, defaults)
        dimensions.append(
            {
                "name": column,
                "display_name": column,
                "weight": default_weight(column, defaults),
                "type": "numeric" if is_numeric_dimension(df[column]) else "categorical",
                "enabled": default_enabled(column, defaults),
                "ease": ease_config["ease"],
                "ease_comments": ease_config.get("measurement_comments", ""),
                "image_description": ease_config.get("image_description", ""),
                "valid_count": valid_count,
                "unique_count": unique_count,
            }
        )
    return dimensions


def default_weight(column: str, defaults: dict[str, dict[str, str]] | None = None) -> float:
    return dimension_default_weight(column, defaults)


def default_enabled(column: str, defaults: dict[str, dict[str, str]] | None = None) -> bool:
    return dimension_default_enabled(column, defaults)


def entropy(labels: pd.Series) -> float:
    total = len(labels)
    if total == 0:
        return 0
    counts = Counter(labels)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def information_gain(df: pd.DataFrame, feature: str, target_col: str) -> float:
    base = entropy(df[target_col])
    total = len(df)
    remainder = 0.0
    for _, group in df.groupby(feature):
        remainder += (len(group) / total) * entropy(group[target_col])
    return base - remainder


def numeric_split_gain(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
) -> tuple[float, float | None]:
    numeric = pd.to_numeric(df[feature], errors="coerce")
    valid_df = df[numeric.notna()].copy()
    valid_numeric = numeric[numeric.notna()]
    unique_values = sorted(valid_numeric.unique())
    if len(unique_values) <= 1:
        return 0.0, None

    base = entropy(valid_df[target_col])
    total = len(valid_df)
    best_gain = -1.0
    best_threshold = None
    for left, right in zip(unique_values, unique_values[1:]):
        threshold = (float(left) + float(right)) / 2
        left_labels = valid_df[valid_numeric <= threshold][target_col]
        right_labels = valid_df[valid_numeric > threshold][target_col]
        if len(left_labels) == 0 or len(right_labels) == 0:
            continue
        remainder = (len(left_labels) / total) * entropy(left_labels)
        remainder += (len(right_labels) / total) * entropy(right_labels)
        gain = base - remainder
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
    return max(best_gain, 0.0), best_threshold


def majority_prediction(df: pd.DataFrame, target_col: str) -> tuple[str, float]:
    counts = Counter(df[target_col])
    label, count = counts.most_common(1)[0]
    return str(label), count / len(df)


def top_target_predictions(
    df: pd.DataFrame,
    target_col: str,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Rank distinct target values by row frequency; attach one representative row's fields as details."""
    if len(df) == 0:
        return []
    counts = Counter(df[target_col])
    total = len(df)
    out: list[dict[str, Any]] = []
    for rank, (label, count) in enumerate(counts.most_common(k), start=1):
        row = df[df[target_col] == label].iloc[0]
        product_info = {
            col: str(row[col])
            for col in df.columns
            if col != target_col and str(row[col]) != MISSING
        }
        out.append(
            {
                "rank": rank,
                "label": str(label),
                "confidence": round(count / total, 4),
                "product_info": product_info,
            }
        )
    return out


def choose_split(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    weights: dict[str, float],
    feature_types: dict[str, str],
) -> dict[str, Any] | None:
    best_split = None
    best_score = -1.0
    for feature in features:
        if feature not in df.columns or df[feature].nunique() <= 1:
            continue
        if feature_types.get(feature) == "numeric":
            gain, threshold = numeric_split_gain(df, feature, target_col)
            if threshold is None:
                continue
            split = {"feature": feature, "split_type": "numeric", "threshold": threshold}
        else:
            gain = information_gain(df, feature, target_col)
            split = {"feature": feature, "split_type": "categorical", "threshold": None}
        score = gain * weights.get(feature, 1.0)
        if score > best_score:
            best_split = split
            best_score = score
    return best_split


def build_tree(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    weights: dict[str, float],
    feature_types: dict[str, str],
) -> TreeNode:
    label, confidence = majority_prediction(df, target_col)
    if len(set(df[target_col])) == 1 or not features:
        return TreeNode(label=label, confidence=confidence, sample_count=len(df))

    split = choose_split(df, features, target_col, weights, feature_types)
    if split is None:
        return TreeNode(label=label, confidence=confidence, sample_count=len(df))

    feature = split["feature"]
    node = TreeNode(
        feature=feature,
        confidence=confidence,
        sample_count=len(df),
        split_type=split["split_type"],
        threshold=split["threshold"],
    )
    remaining = [f for f in features if f != feature]
    if node.split_type == "numeric":
        numeric = pd.to_numeric(df[feature], errors="coerce")
        valid_mask = numeric.notna()
        valid_df = df[valid_mask]
        missing_df = df[~valid_mask]
        vn = pd.to_numeric(valid_df[feature], errors="coerce")
        left_df = valid_df[vn <= node.threshold]
        right_df = valid_df[vn > node.threshold]
        if len(left_df) == 0 or len(right_df) == 0:
            return TreeNode(label=label, confidence=confidence, sample_count=len(df))
        node.children["le"] = build_tree(left_df, remaining, target_col, weights, feature_types)
        node.children["gt"] = build_tree(right_df, remaining, target_col, weights, feature_types)
        if len(missing_df) > 0:
            node.children[MISSING] = build_tree(
                missing_df, remaining, target_col, weights, feature_types
            )
    else:
        for value in sorted_options(df[feature]):
            subset = df[df[feature] == value]
            node.children[str(value)] = build_tree(subset, remaining, target_col, weights, feature_types)
        missing_subset = df[df[feature] == MISSING]
        if len(missing_subset) > 0:
            node.children[MISSING] = build_tree(
                missing_subset, remaining, target_col, weights, feature_types
            )
    return node


def predict_with_tree(node: TreeNode, row: pd.Series) -> tuple[str | None, float]:
    current = node
    while current.feature:
        if current.split_type == "numeric":
            value = pd.to_numeric(pd.Series([row.get(current.feature, MISSING)]), errors="coerce").iloc[0]
            if current.threshold is None:
                return None, current.confidence
            if pd.isna(value):
                child = current.children.get(MISSING)
            else:
                branch = "le" if float(value) <= float(current.threshold) else "gt"
                child = current.children.get(branch)
        else:
            raw = row.get(current.feature, MISSING)
            if raw is None or pd.isna(raw):
                value = MISSING
            else:
                text = str(raw).strip()
                if text == "" or (
                    is_manufacturer_column(str(current.feature)) and text.lower() == "unknown"
                ):
                    value = MISSING
                else:
                    value = text
            child = current.children.get(value)
        if child is None:
            return None, current.confidence
        current = child
    return current.label, current.confidence


def evaluate_tree(tree: TreeNode, df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    correct = 0
    predicted = 0
    details = []
    for _, row in df.iterrows():
        label, confidence = predict_with_tree(tree, row)
        if label is not None:
            predicted += 1
        if label == row[target_col]:
            correct += 1
        actual = str(row[target_col])
        details.append(
            {
                "row_number": int(row.name) + 2,
                "actual": actual,
                "predicted": label,
                "confidence": round(confidence, 4),
                "covered": label is not None,
                "correct": label == actual,
                "name": str(row.get("Name", "")),
                "sku": str(row.get("Castrads SKU", "")),
            }
        )
    total = len(df)
    uncovered = [item for item in details if not item["covered"]]
    incorrect = [item for item in details if item["covered"] and not item["correct"]]
    return {
        "rows": total,
        "predicted_rows": predicted,
        "correct_rows": correct,
        "uncovered_rows": len(uncovered),
        "incorrect_rows": len(incorrect),
        "accuracy": round(correct / total, 4) if total else 0,
        "coverage": round(predicted / total, 4) if total else 0,
        "details": details,
        "uncovered": uncovered,
        "incorrect": incorrect,
    }


def sorted_options(series: pd.Series) -> list[str]:
    values = [str(v) for v in series.drop_duplicates().tolist() if str(v) != MISSING]

    def key(value: str) -> tuple[int, float, str]:
        try:
            return (0, float(value), value)
        except ValueError:
            return (1, 0.0, value.lower())

    return sorted(values, key=key)


def format_threshold(value: float | None) -> str:
    if value is None:
        return ""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def is_numeric_dimension(series: pd.Series) -> bool:
    values = [str(v).strip() for v in series.drop_duplicates().tolist() if str(v) != MISSING]
    if not values:
        return False
    numeric_count = sum(pd.to_numeric(pd.Series([value]), errors="coerce").notna().iloc[0] for value in values)
    return numeric_count / len(values) >= 0.8


def normalize_submitted_value(series: pd.Series, submitted: str) -> str | None:
    submitted = submitted.strip()
    if submitted in set(str(v) for v in series.drop_duplicates().tolist()):
        return submitted

    submitted_number = pd.to_numeric(pd.Series([submitted]), errors="coerce").iloc[0]
    if pd.isna(submitted_number):
        return None

    for value in series.drop_duplicates().tolist():
        value_text = str(value).strip()
        value_number = pd.to_numeric(pd.Series([value_text]), errors="coerce").iloc[0]
        if pd.notna(value_number) and math.isclose(float(value_number), float(submitted_number)):
            return value_text
    return None


def collect_tree_steps(node: TreeNode) -> list[dict[str, Any]]:
    seen = {}

    def walk(n: TreeNode) -> None:
        if n.label or not n.feature:
            return
        if n.feature not in seen:
            seen[n.feature] = {
                "feature": n.feature,
                "split_type": n.split_type or "categorical",
                "threshold": format_threshold(n.threshold),
            }
        for child in n.children.values():
            walk(child)

    walk(node)
    return list(seen.values())


def tree_to_flowchart(node: TreeNode):
    nodes = []
    edges = []
    node_id_counter = [0]

    def traverse(n: TreeNode, level: int = 0, is_root: bool = False):
        current_id = f"node_{node_id_counter[0]}"
        node_id_counter[0] += 1
        if n.label:
            nodes.append(
                {
                    "id": current_id,
                    "type": "result",
                    "text": n.label,
                    "level": level,
                    "is_root": is_root,
                }
            )
        else:
            text = n.feature
            if n.split_type == "numeric":
                text = f"{n.feature} <= {format_threshold(n.threshold)}?"
            nodes.append(
                {
                    "id": current_id,
                    "type": "decision",
                    "text": text,
                    "level": level,
                    "is_root": is_root,
                }
            )
            for value, child in n.children.items():
                child_id = traverse(child, level + 1)
                label = value
                if n.split_type == "numeric":
                    if value == "le":
                        label = f"<= {format_threshold(n.threshold)}"
                    elif value == "gt":
                        label = f"> {format_threshold(n.threshold)}"
                    elif value == MISSING:
                        label = "missing / skipped"
                    else:
                        label = str(value)
                elif value == MISSING:
                    label = "missing / skipped"
                edges.append({"from": current_id, "to": child_id, "label": label})
        return current_id

    traverse(node, is_root=True)
    return nodes, edges
