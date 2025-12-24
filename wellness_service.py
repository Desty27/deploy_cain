from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import sys

import numpy as np
import pandas as pd

# Ensure repo root (contains global_scout/) is importable when running from backend/
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

try:
    from global_scout.src.pipelines.wellness import (  # type: ignore
        load_wellness_data,
        compute_injury_risk,
        summarize_risk,
        generate_coach_guidance,
    )
except Exception:  # pragma: no cover - optional dependency
    load_wellness_data = None  # type: ignore
    compute_injury_risk = None  # type: ignore
    summarize_risk = None  # type: ignore
    generate_coach_guidance = None  # type: ignore


def _local_load_wellness_data(path: Path = Path("global_scout/data/wellness_demo.csv")) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    data = {
        "player_id": ["D001", "D002"],
        "name": ["Demo Bowler", "Demo Batter"],
        "team": ["Australia", "England"],
        "role": ["bowler", "batter"],
        "acute_load": [280, 180],
        "chronic_load": [210, 190],
        "acute_chronic_ratio": [1.33, 0.95],
        "wellness_score": [7.4, 8.5],
        "soreness": [4, 2],
        "sleep_hours": [7.0, 8.2],
        "injury_history": [1, 0],
        "recovery_index": [0.78, 0.86],
        "days_since_last_match": [3, 2],
        "travel_hours": [6, 4],
        "bowling_overs_last_7d": [24, 0],
        "batting_balls_last_7d": [0, 164],
        "sprint_sessions_last_7d": [5, 7],
    }
    return pd.DataFrame(data)


def _local_clip(values, low: float = 0.0, high: float = 1.0, index: Optional[pd.Index] = None) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.clip(lower=low, upper=high)
    arr = np.clip(np.asarray(values, dtype=float), low, high)
    return pd.Series(arr, index=index)


def _local_compute_injury_risk(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    ratio_component = _local_clip((data["acute_chronic_ratio"] - 1.0 + 0.6) / 1.6)
    load_component = _local_clip(data["acute_load"] / 320)
    soreness_component = _local_clip(data["soreness"] / 10)
    wellness_component = _local_clip(1 - data["wellness_score"] / 10)
    sleep_component = _local_clip((8 - data["sleep_hours"]) / 4)
    injury_component = _local_clip(data["injury_history"] / 4)
    recovery_component = _local_clip(1 - data["recovery_index"])
    travel_component = _local_clip(data["travel_hours"] / 14)
    bowling_component = _local_clip(
        np.where(data["role"].str.contains("bowler"), data["bowling_overs_last_7d"] / 28, 0), index=data.index
    )
    batting_component = _local_clip(
        np.where(data["role"].str.contains("batter|keeper"), data["batting_balls_last_7d"] / 180, 0), index=data.index
    )
    sprint_component = _local_clip(data["sprint_sessions_last_7d"] / 8)

    weights = {
        "acute_ratio": 0.18,
        "load": 0.12,
        "soreness": 0.12,
        "wellness": 0.10,
        "sleep": 0.08,
        "injury_history": 0.10,
        "recovery": 0.10,
        "travel": 0.05,
        "bowling": 0.07,
        "batting": 0.04,
        "sprint": 0.04,
    }

    components = {
        "acute_ratio": ratio_component,
        "load": load_component,
        "soreness": soreness_component,
        "wellness": wellness_component,
        "sleep": sleep_component,
        "injury_history": injury_component,
        "recovery": recovery_component,
        "travel": travel_component,
        "bowling": bowling_component,
        "batting": batting_component,
        "sprint": sprint_component,
    }

    risk_raw = pd.Series(0.0, index=data.index)
    for name, comp in components.items():
        risk_raw = risk_raw + (weights[name] * comp)
    risk_score = _local_clip(risk_raw)
    readiness = 1.0 - risk_score

    def classify(score: float) -> str:
        if score >= 0.7:
            return "High"
        if score >= 0.45:
            return "Moderate"
        return "Low"

    driver_labels = []
    for idx, row in pd.DataFrame(components).iterrows():
        contribs = {k: row[k] * weights[k] for k in weights}
        top = max(contribs.items(), key=lambda kv: kv[1])[0]
        driver_labels.append(top)

    recs = []
    for pos, (_, row) in enumerate(data.iterrows()):
        risk = float(risk_score.iloc[pos])
        role = row.get("role", "").lower()
        soreness = row.get("soreness", 0)
        if risk >= 0.7:
            if "bowl" in role:
                rec = "Flag for medical review; cap spells at 2 overs and schedule cryotherapy"
            elif "bat" in role:
                rec = "High risk: lighten net sessions, prioritize recovery modalities"
            else:
                rec = "Restrict high-intensity loads, add physiotherapy screening"
        elif risk >= 0.45:
            if soreness >= 5:
                rec = "Moderate risk: active recovery + compression, monitor before match"
            else:
                rec = "Taper next session by 20%, reinforce sleep and hydration"
        else:
            rec = "Maintain plan; focus on neuromuscular primer and mobility"
        recs.append(rec)

    data["risk_score"] = risk_score.round(3)
    data["readiness"] = readiness.round(3)
    data["risk_level"] = risk_score.apply(classify)
    data["primary_driver"] = driver_labels
    data["recommendation"] = recs
    return data


def _local_summarize_risk(risk_df: pd.DataFrame) -> Dict[str, float]:
    return {
        "mean_risk": float(risk_df["risk_score"].mean()),
        "high_risk_pct": float((risk_df["risk_level"] == "High").mean()),
        "moderate_risk_pct": float((risk_df["risk_level"] == "Moderate").mean()),
        "readiness_mean": float(risk_df["readiness"].mean()),
    }


def _local_generate_coach_guidance(risk_df: pd.DataFrame, match_row: Optional[pd.Series] = None, top_n: int = 3) -> list[str]:
    notes: list[str] = []
    match_context: Dict[str, str] = {}
    if match_row is not None and not match_row.empty:
        match_context = {
            "team_a": str(match_row.get("team_a", "Team A")),
            "team_b": str(match_row.get("team_b", "Team B")),
            "venue": str(match_row.get("venue", "Unknown venue")),
            "date": str(match_row.get("date", "TBD")),
        }
    if match_row is not None and not match_row.empty:
        teams = [match_row.get("team_a"), match_row.get("team_b")]
        for team in teams:
            team_players = risk_df[(risk_df["team"] == team) & (risk_df["risk_level"] == "High")]
            if not team_players.empty:
                names = ", ".join(team_players.head(top_n)["name"].tolist())
                notes.append(f"{team}: Coordinate with Tactical Agent to rotate {names}; limit high-intensity phases and adjust match-ups.")
        if not notes:
            notes.append(f"No high-risk flags for {teams[0]} vs {teams[1]}. Maintain tactical workloads as planned.")
    else:
        notes.append("Integrate with Tactical Agent to align over spells with players showing elevated injury risk.")

    global_high = risk_df[risk_df["risk_level"] == "High"]
    if not global_high.empty:
        top_driver = global_high["primary_driver"].mode().iloc[0]
        notes.append(f"System-wide focus: primary strain driver is {top_driver.replace('_', ' ')} — schedule recovery protocols before next match.")
    else:
        notes.append("Wellness profile stable; continue collaborative monitoring every 24 hours.")

    return notes


class WellnessUnavailable(RuntimeError):
    pass


def _ensure_available() -> None:
    if None in (load_wellness_data, compute_injury_risk, summarize_risk, generate_coach_guidance):
        return


def load_dataset() -> pd.DataFrame:
    _ensure_available()
    if load_wellness_data is not None:
        df = load_wellness_data()  # type: ignore[misc]
    else:
        df = _local_load_wellness_data()
    if df is None or df.empty:
        raise ValueError("Wellness dataset empty; upload or configure data")
    return df


def risk_summary(df: pd.DataFrame) -> Dict[str, Any]:
    _ensure_available()
    risk_df = compute_injury_risk(df) if compute_injury_risk is not None else _local_compute_injury_risk(df)  # type: ignore[misc]
    summary = summarize_risk(risk_df) if summarize_risk is not None else _local_summarize_risk(risk_df)  # type: ignore[misc]
    return {
        "risk_table": risk_df.to_dict("records"),
        "summary": summary,
    }


def guidance(df: pd.DataFrame, match_row: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _ensure_available()
    # Ensure risk fields (risk_level, readiness, etc.) are present
    df = compute_injury_risk(df) if compute_injury_risk is not None else _local_compute_injury_risk(df)  # type: ignore[misc]
    # Convert dict to Series so downstream code can safely use .empty
    match_row_series = None
    if match_row:
        try:
            match_row_series = pd.Series(match_row)
        except Exception:
            match_row_series = None
    notes = (
        generate_coach_guidance(df, match_row=match_row_series)  # type: ignore[misc]
        if generate_coach_guidance is not None
        else _local_generate_coach_guidance(df, match_row=match_row_series)
    )
    return {"guidance": notes}
