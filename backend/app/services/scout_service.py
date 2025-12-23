from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure repo root (contains global_scout/) is importable when running from backend/
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

import pandas as pd

# Attempt to import the Global Scout pipelines; fall back gracefully if absent
try:
    from global_scout.src.pipelines.rank_candidates import score_candidates, mitigate_and_rank  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    score_candidates = None  # type: ignore
    mitigate_and_rank = None  # type: ignore


class ScoutUnavailable(RuntimeError):
    pass


def _ensure_available() -> None:
    if score_candidates is None or mitigate_and_rank is None:
        raise ScoutUnavailable("Global Scout pipeline not available; install global_scout dependencies.")


def load_demo_candidates() -> pd.DataFrame:
    demo_path = Path("global_scout/data/demo_players.csv")
    if demo_path.exists():
        return pd.read_csv(demo_path)
    rng = random.Random(42)
    roles = ["batter", "bowler", "allrounder", "keeper"]
    regions = ["North", "South", "East", "West"]
    rows = []
    for i in range(40):
        role = rng.choice(roles)
        rows.append(
            {
                "player_id": f"D{i+1:03d}",
                "name": f"Demo Player {i+1}",
                "age": rng.randint(18, 34),
                "role": role,
                "league_level": rng.randint(1, 5),
                "matches": rng.randint(5, 70),
                "batting_sr": round(rng.uniform(85, 185), 1) if role != "bowler" else None,
                "batting_avg": round(rng.uniform(12, 62), 1) if role != "bowler" else None,
                "bowling_eco": round(rng.uniform(4.5, 10.5), 2) if role != "batter" else None,
                "bowling_avg": round(rng.uniform(12, 55), 1) if role != "batter" else None,
                "fielding_eff": round(rng.uniform(0.6, 0.98), 2),
                "recent_form": round(rng.uniform(0.2, 0.95), 2),
                "region": rng.choice(regions),
            }
        )
    return pd.DataFrame(rows)


def rank_candidates(
    df: pd.DataFrame,
    protected: str = "region",
    shortlist_k: int = 10,
) -> Dict[str, Any]:
    _ensure_available()
    scored = score_candidates(df)  # type: ignore[misc]
    ranked, audits = mitigate_and_rank(scored, protected=protected, shortlist_k=shortlist_k)  # type: ignore[misc]
    return {
        "ranked": ranked.to_dict("records"),
        "audits": audits,
    }


def rank_from_payload(payload: Dict[str, Any], protected: str, shortlist_k: int) -> Dict[str, Any]:
    df = pd.DataFrame(payload)
    if df.empty:
        raise ValueError("No candidate rows provided")
    return rank_candidates(df=df, protected=protected, shortlist_k=shortlist_k)


def rank_demo(protected: str, shortlist_k: int) -> Dict[str, Any]:
    df = load_demo_candidates()
    return rank_candidates(df=df, protected=protected, shortlist_k=shortlist_k)
