from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, HTTPException, Query

from app.services.wellness_service import WellnessUnavailable, guidance, load_dataset, risk_summary

router = APIRouter()


@router.get("/snapshot")
def wellness_snapshot(use_demo: bool = Query(False, description="Use built-in demo wellness data")) -> Dict[str, Any]:
    """Return injury risk snapshot using the configured wellness pipeline."""
    try:
        df = load_dataset(use_demo=use_demo)
        return risk_summary(df)
    except WellnessUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/guidance")
def wellness_guidance(
    match_context: Optional[Dict[str, Any]] = Body(None),
    use_demo: bool = Query(False, description="Use built-in demo wellness data"),
) -> Dict[str, Any]:
    try:
        df = load_dataset(use_demo=use_demo)
        return guidance(df, match_row=match_context)
    except WellnessUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
