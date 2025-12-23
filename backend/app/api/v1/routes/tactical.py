from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.services.analysis_service import analyze_match, list_matches

router = APIRouter()


@router.get("/matches")
def get_matches() -> dict:
    """List matches available for tactical analysis."""
    return {"matches": list_matches()}


@router.get("/matches/{match_id}")
def get_match_analysis(
    match_id: int,
    innings: Optional[List[int]] = Query(None, description="Optional innings filters e.g. 1&innings=2"),
    monte_trials: int = Query(800, ge=100, le=5000),
) -> dict:
    """Return tactical analysis payload for a match."""
    try:
        return analyze_match(match_id=match_id, innings_filter=innings, monte_trials=monte_trials)
    except FileNotFoundError as exc:  # pragma: no cover - propagated to client
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
