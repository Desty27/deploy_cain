from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from global_scout.src.services.azure_openai import generate_medical_coordination_notes

DEFAULT_DATA_PATH = Path("global_scout/data/wellness_demo.csv")


def load_wellness_data(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load wellness data from CSV. Generates a small demo frame if file missing."""
    if path.exists():
        return pd.read_csv(path)

    # Fallback synthetic frame (mirrors schema)
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


def _component_clip(values, low: float = 0.0, high: float = 1.0, index: Optional[pd.Index] = None) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.clip(lower=low, upper=high)
    arr = np.clip(np.asarray(values, dtype=float), low, high)
    return pd.Series(arr, index=index)


def compute_injury_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Compute injury risk score, readiness, risk level, and primary driver."""
    data = df.copy()

    ratio_component = _component_clip((data["acute_chronic_ratio"] - 1.0 + 0.6) / 1.6)
    load_component = _component_clip(data["acute_load"] / 320)
    soreness_component = _component_clip(data["soreness"] / 10)
    wellness_component = _component_clip(1 - data["wellness_score"] / 10)
    sleep_component = _component_clip((8 - data["sleep_hours"]) / 4)
    injury_component = _component_clip(data["injury_history"] / 4)
    recovery_component = _component_clip(1 - data["recovery_index"])
    travel_component = _component_clip(data["travel_hours"] / 14)
    bowling_component = _component_clip(
        np.where(data["role"].str.contains("bowler"), data["bowling_overs_last_7d"] / 28, 0),
        index=data.index,
    )
    batting_component = _component_clip(
        np.where(data["role"].str.contains("batter|keeper"), data["batting_balls_last_7d"] / 180, 0),
        index=data.index,
    )
    sprint_component = _component_clip(data["sprint_sessions_last_7d"] / 8)

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
    risk_score = _component_clip(risk_raw)
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
        rec: str
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


def summarize_risk(risk_df: pd.DataFrame) -> Dict[str, float]:
    summary = {
        "mean_risk": float(risk_df["risk_score"].mean()),
        "high_risk_pct": float((risk_df["risk_level"] == "High").mean()),
        "moderate_risk_pct": float((risk_df["risk_level"] == "Moderate").mean()),
        "readiness_mean": float(risk_df["readiness"].mean()),
    }
    return summary


def generate_coach_guidance(
    risk_df: pd.DataFrame,
    match_row: Optional[pd.Series] = None,
    top_n: int = 3,
) -> List[str]:
    notes: List[str] = []
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
                notes.append(
                    f"{team}: Coordinate with Tactical Agent to rotate {names}; limit high-intensity phases and adjust match-ups."
                )
        if not notes:
            notes.append(
                f"No high-risk flags for {teams[0]} vs {teams[1]}. Maintain tactical workloads as planned."
            )
    else:
        notes.append(
            "Integrate with Tactical Agent to align over spells with players showing elevated injury risk."
        )

    global_high = risk_df[risk_df["risk_level"] == "High"]
    if not global_high.empty:
        top_driver = global_high["primary_driver"].mode().iloc[0]
        notes.append(
            f"System-wide focus: primary strain driver is {top_driver.replace('_', ' ')} — schedule recovery protocols before next match."
        )
    else:
        notes.append("Wellness profile stable; continue collaborative monitoring every 24 hours.")

    # Build roster snapshot for LLM enrichment
    high_risk = (
        risk_df[risk_df["risk_level"] == "High"]
        .sort_values("risk_score", ascending=False)
        .head(top_n)
    )
    high_risk_report = "; ".join(
        f"{row['name']} ({row['team']}, driver {row['primary_driver']}, risk {row['risk_score']:.2f})"
        for _, row in high_risk.iterrows()
    ) or "None"

    moderate_risk = (
        risk_df[risk_df["risk_level"] == "Moderate"]
        .sort_values("risk_score", ascending=False)
        .head(top_n)
    )
    moderate_risk_report = "; ".join(
        f"{row['name']} ({row['team']}, driver {row['primary_driver']}, risk {row['risk_score']:.2f})"
        for _, row in moderate_risk.iterrows()
    ) or "None"

    workload_flags: List[str] = []
    ac_spikes = risk_df[risk_df["acute_chronic_ratio"] > 1.3].head(top_n)
    if not ac_spikes.empty:
        names = ", ".join(ac_spikes["name"].tolist())
        workload_flags.append(f"AC ratio spike >1.3 for {names}")

    acute_vs_chronic = risk_df[risk_df["acute_load"] > (risk_df["chronic_load"] * 1.25)].head(top_n)
    if not acute_vs_chronic.empty:
        names = ", ".join(acute_vs_chronic["name"].tolist())
        workload_flags.append(f"Acute load >125% of chronic for {names}")

    travel_stress = risk_df[risk_df.get("travel_hours", 0) >= 8]
    if not travel_stress.empty:
        names = ", ".join(travel_stress.head(top_n)["name"].tolist())
        workload_flags.append(f"Long-haul travel flagged for {names}")

    roster_snapshot = {
        "high_risk_report": high_risk_report,
        "moderate_risk_report": moderate_risk_report,
        "readiness_mean": float(risk_df["readiness"].mean()),
        "workload_flags": "; ".join(workload_flags) if workload_flags else "None",
    }

    # Add per-team summaries so the LLM can produce team-specific guidance
    per_team_high = {}
    per_team_moderate = {}
    per_team_workload = {}
    teams = sorted(risk_df["team"].dropna().unique())
    for t in teams:
        high = risk_df[(risk_df["team"] == t) & (risk_df["risk_level"] == "High")]
        if not high.empty:
            per_team_high[t] = "; ".join(high.head(top_n).apply(lambda r: f"{r['name']} (risk {r['risk_score']:.2f})", axis=1).tolist())
        else:
            per_team_high[t] = "None"

        mod = risk_df[(risk_df["team"] == t) & (risk_df["risk_level"] == "Moderate")]
        if not mod.empty:
            per_team_moderate[t] = "; ".join(mod.head(top_n).apply(lambda r: f"{r['name']} (risk {r['risk_score']:.2f})", axis=1).tolist())
        else:
            per_team_moderate[t] = "None"

        # workload snapshot: AC spikes and long travel per team
        flags = []
        t_ac = risk_df[(risk_df["team"] == t) & (risk_df["acute_chronic_ratio"] > 1.3)]
        if not t_ac.empty:
            flags.append("AC>1.3: " + ", ".join(t_ac.head(top_n)["name"].tolist()))
        t_travel = risk_df[(risk_df["team"] == t) & (risk_df.get("travel_hours", 0) >= 8)]
        if not t_travel.empty:
            flags.append("Travel: " + ", ".join(t_travel.head(top_n)["name"].tolist()))
        per_team_workload[t] = "; ".join(flags) if flags else "None"

    roster_snapshot["per_team_high"] = per_team_high
    roster_snapshot["per_team_moderate"] = per_team_moderate
    roster_snapshot["per_team_workload"] = per_team_workload

    llm_notes = generate_medical_coordination_notes(match_context, roster_snapshot, notes)
    if llm_notes:
        return llm_notes

    return notes
