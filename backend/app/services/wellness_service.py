from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import sys

# Ensure repo root (contains global_scout/) is importable when running from backend/
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

import pandas as pd

try:
    from global_scout.src.pipelines.wellness import (  # type: ignore
        load_wellness_data,
        compute_injury_risk,
        summarize_risk,
        generate_coach_guidance,
    )
except Exception:  # pragma: no cover - optional dependency
    load_wellness_data = None  # type: ignore
    compute_injury_risk = None  # type: ignore
    summarize_risk = None  # type: ignore
    generate_coach_guidance = None  # type: ignore


class WellnessUnavailable(RuntimeError):
    pass


def _ensure_available() -> None:
    if None in (load_wellness_data, compute_injury_risk, summarize_risk, generate_coach_guidance):
        raise WellnessUnavailable("Wellness pipeline not available; install global_scout dependencies.")


def load_dataset() -> pd.DataFrame:
    _ensure_available()
    df = load_wellness_data()  # type: ignore[misc]
    if df is None or df.empty:
        raise ValueError("Wellness dataset empty; upload or configure data")
    return df


def risk_summary(df: pd.DataFrame) -> Dict[str, Any]:
    _ensure_available()
    risk_df = compute_injury_risk(df)  # type: ignore[misc]
    summary = summarize_risk(risk_df)  # type: ignore[misc]
    return {
        "risk_table": risk_df.to_dict("records"),
        "summary": summary,
    }


def guidance(df: pd.DataFrame, match_row: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _ensure_available()
    # Ensure risk fields (risk_level, readiness, etc.) are present
    df = compute_injury_risk(df)  # type: ignore[misc]
    # Convert dict to Series so downstream code can safely use .empty
    match_row_series = None
    if match_row:
        try:
            match_row_series = pd.Series(match_row)
        except Exception:
            match_row_series = None
    notes = generate_coach_guidance(df, match_row=match_row_series)  # type: ignore[misc]
    return {"guidance": notes}
