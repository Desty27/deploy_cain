from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
import pandas as pd
from io import BytesIO
from typing import Optional
from ..pipelines.rank_candidates import score_candidates, mitigate_and_rank

app = FastAPI(title="Global Scout API", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/rank")
async def rank_candidates(
    file: UploadFile = File(..., description="CSV with candidate rows"),
    protected: str = Query("region"),
    shortlist_k: int = Query(10, ge=1, le=200)
):
    content = await file.read()
    df = pd.read_csv(BytesIO(content))
    df = score_candidates(df)
    ranked, audits = mitigate_and_rank(df, protected=protected, shortlist_k=shortlist_k)
    return JSONResponse(
        {
            "audits": audits,
            "candidates": ranked.to_dict(orient="records"),
        }
    )
