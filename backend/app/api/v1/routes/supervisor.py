from typing import Any, Dict

from fastapi import APIRouter, Query

from app.services.supervisor_service import build_supervisor

router = APIRouter()


@router.get("/matches/{match_id}")
def supervisor_summary(
    match_id: int,
    monte_trials: int = Query(800, ge=100, le=5000),
    overs_window: int = Query(3, ge=2, le=6),
) -> Dict[str, Any]:
    """Aggregate tactical, wellness, integrity, and scouting snapshots for the Head Coach view."""
    return build_supervisor(match_id=match_id, monte_trials=monte_trials, overs_window=overs_window)
