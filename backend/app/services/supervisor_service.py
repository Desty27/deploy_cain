from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from app.services import analysis_service
from app.services.data_loader import DatasetMissingError, load_frames
from app.services.integrity_service import IntegrityUnavailable, analyze as integrity_analyze
from app.services.scout_service import ScoutUnavailable, rank_demo
from app.services.wellness_service import WellnessUnavailable, load_dataset, risk_summary


def _phase_chart(phase_team: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for team, phases in (phase_team or {}).items():
        for phase_name, stats in phases.items():
            try:
                rows.append({"team": team, "phase": phase_name, "rpo": round(float(stats.get("rpo", 0)), 2)})
            except Exception:
                continue
    return rows


def _trajectory(over_summary: pd.DataFrame) -> List[Dict[str, Any]]:
    if over_summary.empty:
        return []
    cumulative = over_summary.sort_values(["innings", "over"]).copy()
    cumulative["cumulative_runs"] = cumulative.groupby("innings")["runs"].cumsum()
    return [
        {
            "innings": int(row["innings"]),
            "over": int(row["over"]),
            "runs": int(row["runs"]),
            "cumulative_runs": int(row["cumulative_runs"]),
        }
        for _, row in cumulative.iterrows()
    ]


def _pressure_chart(pressure_windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in pressure_windows[:8]:
        try:
            rows.append(
                {
                    "innings": int(row.get("innings", 0)),
                    "window": f"{int(row.get('start_over', 0))}-{int(row.get('end_over', 0))}",
                    "pressure_index": float(row.get("pressure_index", 0)),
                }
            )
        except Exception:
            continue
    return rows


def _wellness_risk_chart(risk_table: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not risk_table:
        return []
    sorted_rows = sorted(risk_table, key=lambda r: float(r.get("risk_score", 0) or 0), reverse=True)
    top_rows = sorted_rows[:8]
    return [
        {
            "name": row.get("name", "Player"),
            "team": row.get("team"),
            "risk_score": float(row.get("risk_score", 0) or 0),
            "readiness": float(row.get("readiness", 0) or 0),
        }
        for row in top_rows
    ]


def _mission_brief(
    winner: Optional[str],
    win_first: Optional[float],
    win_chase: Optional[float],
    readiness_mean: Optional[float],
    integrity_peak: Optional[float],
) -> str:
    parts: List[str] = []
    if winner:
        parts.append(f"Match result: {winner} (historical).")
    if win_first is not None and win_chase is not None:
        parts.append(f"Simulation split → Batting first {win_first:.0%}, Chasing {win_chase:.0%}.")
    if readiness_mean is not None:
        parts.append(f"Squad readiness mean {readiness_mean:.2f} — align overs with fitter units.")
    if integrity_peak is not None:
        parts.append(f"Peak decision-pressure index {integrity_peak:.2f}; prep third umpire cues.")
    if not parts:
        return "Supervisor brief unavailable — provide match data to generate signals."
    return " ".join(parts)


def _callouts(
    tactical: Optional[List[str]],
    wellness: Optional[List[str]],
    integrity: Optional[List[str]],
    scout_top: Optional[List[str]],
) -> Dict[str, List[str]]:
    return {
        "tactical": tactical or [],
        "wellness": wellness or [],
        "integrity": integrity or [],
        "recruiting": scout_top or [],
    }


def _top_candidates(ranked: List[Dict[str, Any]], limit: int = 5) -> List[str]:
    if not ranked:
        return []
    picks = ranked[:limit]
    labels: List[str] = []
    for row in picks:
        name = row.get("name", "Candidate")
        role = row.get("role", "role")
        labels.append(f"{name} ({role})")
    return labels


def build_supervisor(match_id: int, overs_window: int = 3, monte_trials: int = 800) -> Dict[str, Any]:
    try:
        deliveries_df, matches_df, _ = load_frames()
    except DatasetMissingError as exc:
        raise FileNotFoundError(str(exc))

    match_row_df = matches_df[matches_df.get("match_id") == match_id]
    if match_row_df.empty:
        raise FileNotFoundError(f"match_id {match_id} not found in matches.csv")
    match_row = match_row_df.iloc[0]

    match_deliveries = deliveries_df[deliveries_df.get("match_id") == match_id].copy()
    if match_deliveries.empty:
        raise ValueError("No deliveries found for match")

    runs_by_innings = match_deliveries.groupby("innings")["runs_total"].sum().astype(int).to_dict()
    wickets_by_innings = match_deliveries.groupby("innings")["wicket"].sum().astype(int).to_dict()

    analysis: Dict[str, Any] = {}
    integrity: Dict[str, Any] = {}
    wellness: Dict[str, Any] = {}
    scout: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    try:
        analysis = analysis_service.analyze_match(match_id=match_id, monte_trials=monte_trials)
    except Exception as exc:
        errors["analysis_error"] = str(exc)

    try:
        integrity = integrity_analyze(match_id=match_id, overs_window=overs_window)
    except (IntegrityUnavailable, FileNotFoundError, ValueError) as exc:
        errors["integrity_error"] = str(exc)

    try:
        df = load_dataset()
        wellness = risk_summary(df)
    except (WellnessUnavailable, ValueError) as exc:
        errors["wellness_error"] = str(exc)

    try:
        scout = rank_demo(protected="region", shortlist_k=10)
    except (ScoutUnavailable, ValueError) as exc:
        errors["scout_error"] = str(exc)

    over_summary_df = analysis_service.compute_over_summary(match_deliveries)
    phase_chart = _phase_chart(analysis.get("phase_team") if analysis else {})
    trajectory = _trajectory(over_summary_df)
    pressure_chart = _pressure_chart(integrity.get("pressure_windows", []) if integrity else [])
    wellness_chart = _wellness_risk_chart(wellness.get("risk_table", []) if wellness else [])

    win_first = None
    win_chase = None
    monte = analysis.get("monte_carlo") if analysis else None
    if monte:
        try:
            win_first = float(monte.get("win_prob_innings1", 0))
            win_chase = float(monte.get("win_prob_innings2", 0))
        except Exception:
            win_first = win_chase = None

    readiness_mean = None
    if wellness:
        try:
            readiness_mean = float(wellness.get("summary", {}).get("readiness_mean"))
        except Exception:
            readiness_mean = None

    peak_pressure = None
    if integrity:
        try:
            peak_pressure = float(integrity.get("peak_pressure_index", 0))
        except Exception:
            peak_pressure = None

    mission_brief = _mission_brief(
        winner=analysis.get("summary", {}).get("winner") if analysis else None,
        win_first=win_first,
        win_chase=win_chase,
        readiness_mean=readiness_mean,
        integrity_peak=peak_pressure,
    )

    callouts = _callouts(
        tactical=analysis.get("recommendations") if analysis else None,
        wellness=wellness.get("guidance") if wellness else None,
        integrity=integrity.get("alerts") if integrity else None,
        scout_top=_top_candidates(scout.get("ranked", []) if scout else []),
    )

    response: Dict[str, Any] = {
        "match_meta": {
            "match_id": match_id,
            "title": analysis.get("match", {}).get("title") if analysis else None,
            "winner": analysis.get("match", {}).get("winner") if analysis else None,
            "team_a": analysis.get("match", {}).get("team_a") if analysis else match_row.get("team_a"),
            "team_b": analysis.get("match", {}).get("team_b") if analysis else match_row.get("team_b"),
            "venue": match_row.get("venue"),
            "date": match_row.get("date"),
            "runs_by_innings": {str(k): int(v) for k, v in runs_by_innings.items()},
            "wickets_by_innings": {str(k): int(v) for k, v in wickets_by_innings.items()},
        },
        "tactical": analysis,
        "integrity": integrity,
        "wellness": wellness,
        "scout": scout,
        "mission_brief": mission_brief,
        "callouts": callouts,
        "charts": {
            "phase": phase_chart,
            "trajectory": trajectory,
            "pressure": pressure_chart,
            "wellness": wellness_chart,
        },
        "shared_data_note": "Shared uploads not yet wired in this API; ensure deliveries/matches CSVs are present and wellness/candidate data is available.",
        "overs_window": overs_window,
        **errors,
    }

    return response
