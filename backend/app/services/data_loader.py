from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple
import sys

import pandas as pd

from app.core.config import get_settings

# Ensure repository root (contains src/) is on sys.path so imports work when running inside deploy_cain/backend
repo_root = Path(__file__).resolve().parents[4]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from src.analyzer import load_deliveries_csv, load_matches_csv, load_players_csv


class DatasetMissingError(FileNotFoundError):
    """Raised when expected CSV assets are missing."""


@lru_cache(maxsize=1)
def _load_raw_frames() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, str]]:
    settings = get_settings()
    base = Path(settings.data_dir)
    deliveries_path = base / "deliveries.csv"
    matches_path = base / "matches.csv"
    players_path = base / "players.csv"

    if not deliveries_path.exists() or not matches_path.exists():
        raise DatasetMissingError("deliveries.csv or matches.csv not found in data_dir")

    deliveries = pd.DataFrame(load_deliveries_csv(deliveries_path))
    matches = pd.DataFrame(load_matches_csv(matches_path))
    players = load_players_csv(players_path)

    if not deliveries.empty:
        deliveries["runs_off_bat"] = deliveries["runs_off_bat"].fillna(0)
        deliveries["extras"] = deliveries["extras"].fillna(0)
        deliveries["runs_total"] = deliveries["runs_off_bat"] + deliveries["extras"]
        deliveries["wicket"] = deliveries["dismissed_batter_id"].notna()
    if not matches.empty and "match_id" in matches.columns:
        matches["match_id"] = matches["match_id"].apply(lambda v: int(v) if pd.notna(v) else None)
    players_map = {int(pid): name for pid, name in players.items()} if players else {}
    return deliveries, matches, players_map


def load_frames() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, str]]:
    return _load_raw_frames()
