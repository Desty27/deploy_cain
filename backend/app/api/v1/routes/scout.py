from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, HTTPException, Query

from app.services.scout_service import ScoutUnavailable, rank_demo, rank_from_payload

router = APIRouter()


@router.post("/rank")
def rank_candidates(
    payload: Optional[Dict[str, Any]] = Body(None, description="Candidate rows as list of objects under key 'rows'"),
    use_demo: bool = Query(False, description="Use built-in demo dataset instead of payload"),
    protected: str = Query("region", description="Protected attribute column"),
    shortlist_k: int = Query(10, ge=1, le=100),
) -> Dict[str, Any]:
    """Run fairness-aware ranking via Global Scout."""
    try:
        if use_demo or payload is None:
            return rank_demo(protected=protected, shortlist_k=shortlist_k)
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if rows is None:
            raise ValueError("Payload must contain 'rows' list")
        return rank_from_payload(rows, protected=protected, shortlist_k=shortlist_k)
    except ScoutUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
