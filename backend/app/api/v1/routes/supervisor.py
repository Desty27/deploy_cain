from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from app.services import analysis_service
from app.services.integrity_service import IntegrityUnavailable, analyze as integrity_analyze
from app.services.scout_service import ScoutUnavailable, rank_demo
from app.services.wellness_service import WellnessUnavailable, risk_summary, load_dataset

router = APIRouter()


@router.get("/matches/{match_id}")
def supervisor_summary(match_id: int, monte_trials: int = Query(800, ge=100, le=5000)) -> Dict[str, Any]:
    """Aggregate tactical, wellness, integrity, and scouting snapshots."""
    response: Dict[str, Any] = {}
    try:
        response["analysis"] = analysis_service.analyze_match(match_id=match_id, monte_trials=monte_trials)
    except Exception as exc:  # pragma: no cover - best effort aggregation
        response["analysis_error"] = str(exc)

    try:
        response["integrity"] = integrity_analyze(match_id=match_id, overs_window=3)
    except (IntegrityUnavailable, FileNotFoundError, ValueError) as exc:
        response["integrity_error"] = str(exc)

    try:
        df = load_dataset()
        response["wellness"] = risk_summary(df)
    except (WellnessUnavailable, ValueError) as exc:
        response["wellness_error"] = str(exc)

    try:
        response["scout"] = rank_demo(protected="region", shortlist_k=10)
    except (ScoutUnavailable, ValueError) as exc:
        response["scout_error"] = str(exc)

    return response
