from fastapi import APIRouter

from app.api.v1.routes import analysis, health, integrity, scout, supervisor, wellness, tactical

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
api_router.include_router(tactical.router, prefix="/tactical", tags=["tactical"])
api_router.include_router(scout.router, prefix="/scout", tags=["scout"])
api_router.include_router(wellness.router, prefix="/wellness", tags=["wellness"])
api_router.include_router(integrity.router, prefix="/integrity", tags=["integrity"])
api_router.include_router(supervisor.router, prefix="/supervisor", tags=["supervisor"])
