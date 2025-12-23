from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, HTTPException

from app.services.wellness_service import WellnessUnavailable, guidance, load_dataset, risk_summary

router = APIRouter()


@router.get("/live")
def live() -> dict:
    """Liveness probe for Render/health checks."""
    return {"status": "ok"}


@router.get("/ready")
def ready() -> dict:
    """Readiness probe indicating dependencies are loaded."""
    return {"status": "ready"}


@router.get("/wellness")
def wellness_snapshot() -> Dict[str, Any]:
    """Alias for the wellness snapshot so Performance & Wellness can be reached via /health."""
    try:
        df = load_dataset()
        return risk_summary(df)
    except WellnessUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/wellness/guidance")
def wellness_guidance(match_context: Optional[Dict[str, Any]] = Body(None)) -> Dict[str, Any]:
    """Alias for generating wellness guidance via /health."""
    try:
        df = load_dataset()
        return guidance(df, match_row=match_context)
    except WellnessUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
