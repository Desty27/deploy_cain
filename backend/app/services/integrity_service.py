from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path
import sys

import pandas as pd

from app.services.data_loader import load_frames

# Ensure repo root (contains global_scout/) is importable when running from backend/
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

try:
    from global_scout.src.agents.integrity.adjudication_agent import build_agent  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    build_agent = None  # type: ignore


class IntegrityUnavailable(RuntimeError):
    pass


def analyze(match_id: int, overs_window: int = 3) -> Dict[str, Any]:
    if build_agent is None:
        raise IntegrityUnavailable("Integrity agent unavailable; install global_scout dependencies.")

    deliveries_df, matches_df, players_map = load_frames()
    match_row_df = matches_df[matches_df.get("match_id") == match_id]
    if match_row_df.empty:
        raise FileNotFoundError(f"match_id {match_id} not found")
    match_row = match_row_df.iloc[0]

    match_deliveries = deliveries_df[deliveries_df.get("match_id") == match_id].copy()
    if match_deliveries.empty:
        raise ValueError("No deliveries found for match")

    agent = build_agent(match_deliveries, match_row, overs_window=overs_window)  # type: ignore[misc]
    insights = agent.build_insights()

    return {
        "alerts": insights.alerts,
        # high_verdict_windows is already a list of dicts from the agent
        "high_verdict_windows": insights.high_verdict_windows,
        # pressure_windows is a DataFrame; convert to records if available
        "pressure_windows": insights.pressure_windows.to_dict("records") if not insights.pressure_windows.empty else [],
        "review_hotspots": insights.review_hotspots.to_dict("records") if not insights.review_hotspots.empty else [],
        "narrative": insights.narrative,
    }
