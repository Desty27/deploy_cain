# CAIN Backend (FastAPI)

Production-grade API that mirrors the Streamlit POC capabilities: tactical match analytics, scouting, wellness, integrity, and supervisor aggregation.

## Local development

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Base data is loaded from the repository root (`deliveries.csv`, `matches.csv`, `players.csv`). Override via environment variable `CAIN_DATA_DIR` if needed.

## API surface

- `GET /api/v1/health/live` – liveness
- `GET /api/v1/analysis/matches` – list matches
- `GET /api/v1/analysis/matches/{match_id}` – tactical analysis (phase RPO, wicket clusters, Monte Carlo, ball ledger)
- `POST /api/v1/scout/rank` – bias-aware ranking (payload or demo)
- `GET /api/v1/wellness/snapshot` – injury/readiness snapshot (requires global_scout wellness pipeline)
- `GET /api/v1/integrity/matches/{match_id}` – adjudication pressure windows (optional dependency)
- `GET /api/v1/supervisor/matches/{match_id}` – aggregated multi-agent snapshot

## Render deployment

Use `render.yaml` at repo root. Backend service configuration:

- Type: Web Service
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port 10000`
- Build: `pip install -r requirements.txt`
- Root Directory: `deploy_cain/backend`

Expose port `10000` (Render will map). Set environment variables for Azure OpenAI if needed (`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`).
