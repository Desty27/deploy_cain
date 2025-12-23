# Global Scout (Bias-Aware Scouting & Recruitment)

Objective, bias-aware talent identification based on performance, strategic fit, and fairness auditing.

## Quick start (Windows)

1. Install dependencies in your existing env:
   
   - Add these to requirements.txt if missing:
     - fastapi
     - uvicorn
     - numpy
     - scikit-learn (optional)

2. Create demo data
   
   ```powershell
   python global_scout\scripts\seed_demo_data.py
   ```

3. Run ranking pipeline
   
   ```powershell
   python -c "from global_scout.src.pipelines.rank_candidates import run_pipeline; import pathlib; run_pipeline(pathlib.Path('global_scout/data/demo_players.csv'), pathlib.Path('global_scout/data/candidate_rankings.csv'))"
   ```

4. Start API
   
   ```powershell
   uvicorn global_scout.src.api.main:app --reload
   ```

## API
- POST /rank (multipart/form-data) with file=<CSV>; params: protected=region, shortlist_k=10
- GET /health

## Scoring
- Performance score: role-sensitive blend (batter/bowler/allrounder).
- Strategic fit: league level, matches, recent form.
- Final score: within-group standardization (fairness) + monotonic calibration -> ranking.

## Fairness
- Reports demographic parity difference and equal opportunity difference across `protected` (default: region). No protected attribute is used as a positive boost; fairness is applied post-scoring to reduce unintended skews.
