#!/usr/bin/env python3
"""Quick check: terminal state + confidence below threshold -> ambiguous + top candidates."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from flow_runtime import TreeNode, get_next_step_context, session_state


def main() -> None:
    df = pd.DataFrame(
        {
            "SKU": ["A", "B", "C", "A", "B", "C"],
            "Manufacturer": ["X", "X", "X", "Y", "Y", "Y"],
        }
    )
    model = {
        "config": {"target_column": "SKU", "dimensions": [{"name": "Manufacturer", "enabled": True}]},
        "metadata": {"name": "verify"},
        "data": df,
    }
    session_state.clear()
    session_state.update(
        {
            "data": df.copy(),
            "model": model,
            "history": [],
            "actions": [],
            "asked_features": [],
            "skipped_features": [],
            "vision_values": {},
            "vision_notes": "",
            "vision_error": "",
        }
    )
    session_state["node"] = TreeNode(label="A", confidence=2 / 6, sample_count=6)

    status, ctx = get_next_step_context(model, confidence_threshold=0.86)
    assert status == "ambiguous", status
    assert ctx.get("ambiguous") is True
    assert len(ctx["top_candidates"]) == 3
    assert {c["label"] for c in ctx["top_candidates"]} == {"A", "B", "C"}
    print("OK — ambiguous flow returns 3 candidates (each ~33.3%).")


if __name__ == "__main__":
    main()
