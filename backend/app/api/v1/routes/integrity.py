from fastapi import APIRouter, HTTPException, Query

from app.services.integrity_service import IntegrityUnavailable, analyze

router = APIRouter()


@router.get("/matches/{match_id}")
def integrity_for_match(match_id: int, overs_window: int = Query(3, ge=2, le=8)) -> dict:
    try:
        return analyze(match_id=match_id, overs_window=overs_window)
    except IntegrityUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
