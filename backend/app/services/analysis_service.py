from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from app.services.data_loader import DatasetMissingError, load_frames
from src.analyzer import (
    compute_per_bowler_wicket_clusters,
    compute_phase_run_rates,
    compute_phase_run_rates_per_team,
    monte_carlo_simulation,
    plot_phase_run_rates_per_team,
    safe_int,
)


def label_phase(over: int) -> str:
    if over <= 5:
        return "Powerplay"
    if over <= 14:
        return "Middle"
    return "Death"


def compute_innings_summary(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    if df.empty:
        return pd.DataFrame(columns=["innings", "runs", "wickets", "overs", "run_rate", "boundaries"])
    for innings in sorted(df["innings"].dropna().astype(int).unique()):
        inn_df = df[df["innings"] == innings]
        runs = float(inn_df["runs_total"].sum())
        wickets = int(inn_df["wicket"].sum())
        balls = int(inn_df["runs_total"].count())
        overs_completed = balls // 6
        balls_remaining = balls % 6
        overs_text = f"{overs_completed}.{balls_remaining}"
        run_rate = (runs / (balls / 6)) if balls else 0
        boundaries = int((inn_df["runs_off_bat"] == 4).sum() + (inn_df["runs_off_bat"] == 6).sum())
        records.append(
            {
                "innings": innings,
                "runs": int(runs),
                "wickets": wickets,
                "overs": overs_text,
                "run_rate": round(run_rate, 2),
                "boundaries": boundaries,
            }
        )
    return pd.DataFrame.from_records(records)


def compute_over_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["innings", "over", "runs", "wickets"])
    grouped = (
        df.groupby(["innings", "over"])
        .agg(runs=("runs_total", "sum"), wickets=("wicket", "sum"))
        .reset_index()
        .sort_values(["innings", "over"])
    )
    grouped["runs"] = grouped["runs"].astype(int)
    grouped["wickets"] = grouped["wickets"].astype(int)
    return grouped


def compute_wicket_clusters_sliding(df: pd.DataFrame, window: int = 6, top_n: int = 5) -> pd.DataFrame:
    """Sliding-window wicket clusters derived from ball-by-ball data."""
    if df.empty or "wicket" not in df.columns:
        return pd.DataFrame(columns=["innings", "start_over", "end_over", "wickets"])

    working = df.dropna(subset=["innings", "over"]).copy()
    if working.empty:
        return pd.DataFrame(columns=["innings", "start_over", "end_over", "wickets"])

    working["innings"] = working["innings"].astype(int)
    working["over"] = working["over"].astype(int)
    wickets_by_over = working.groupby(["innings", "over"])["wicket"].sum().reset_index()

    clusters: List[Dict[str, int]] = []
    for innings_key, inn_df in wickets_by_over.groupby("innings"):
        innings_val = safe_int(innings_key)
        if innings_val is None:
            continue
        overs_series = inn_df["over"].astype(int)
        overs_list = sorted(overs_series.tolist())
        if not overs_list:
            continue
        min_over = int(overs_list[0])
        max_over = int(overs_list[-1])
        for start in range(min_over, max_over + 1):
            end = start + window - 1
            wickets_sum = inn_df[(inn_df["over"] >= start) & (inn_df["over"] <= end)]["wicket"].sum()
            wickets = int(wickets_sum) if wickets_sum else 0
            if wickets > 0:
                clusters.append(
                    {
                        "innings": int(innings_val),
                        "start_over": int(start),
                        "end_over": int(end),
                        "wickets": wickets,
                    }
                )

    if not clusters:
        return pd.DataFrame(columns=["innings", "start_over", "end_over", "wickets"])

    clusters_df = pd.DataFrame(clusters).drop_duplicates()
    clusters_df = clusters_df.sort_values(["wickets", "innings", "start_over"], ascending=[False, True, True])
    return clusters_df.head(top_n)


def enrich_ball_by_ball(df: pd.DataFrame, players: Dict[int, str]) -> pd.DataFrame:
    if df.empty:
        return df
    enriched = df.copy()
    for column, target in [("batter_id", "batter"), ("bowler_id", "bowler"), ("non_striker_id", "non_striker")]:
        enriched[target] = enriched[column].apply(lambda pid: players.get(int(pid), str(pid)) if not pd.isna(pid) else "")
    enriched["runs_total"] = enriched["runs_total"].astype(int)
    enriched["wicket_type"] = enriched["wicket_type"].fillna("-").replace({"": "-", "None": "-", "none": "-"})
    enriched["dismissed"] = enriched["dismissed_batter_id"].apply(
        lambda pid: players.get(int(pid), str(int(pid))) if not (pid is None or pd.isna(pid)) else "-"
    )
    enriched["phase"] = enriched["over"].apply(lambda o: label_phase(int(o)))
    columns = [
        "innings",
        "over",
        "ball",
        "phase",
        "batter",
        "bowler",
        "runs_off_bat",
        "extras",
        "runs_total",
        "wicket_type",
        "dismissed",
    ]
    return enriched[columns]


def list_matches() -> List[Dict[str, Any]]:
    deliveries_df, matches_df, _ = load_frames()
    if matches_df.empty:
        return []
    matches_df = matches_df.dropna(subset=["match_id"])
    records: List[Dict[str, Any]] = []
    for _, row in matches_df.iterrows():
        mid = safe_int(row.get("match_id"))
        if mid is None:
            continue
        record = {
            "match_id": mid,
            "team_a": row.get("team_a"),
            "team_b": row.get("team_b"),
            "venue": row.get("venue"),
            "date": row.get("date"),
            "winner": row.get("winner"),
            "series_name": row.get("series_name"),
        }
        records.append(record)
    return records


def analyze_match(match_id: int, innings_filter: Optional[List[int]] = None, monte_trials: int = 800) -> Dict[str, Any]:
    try:
        deliveries_df, matches_df, players_map = load_frames()
    except DatasetMissingError as exc:
        raise FileNotFoundError(str(exc))

    matches = matches_df[matches_df.get("match_id") == match_id]
    if matches.empty:
        raise FileNotFoundError(f"match_id {match_id} not found in matches.csv")

    match_row = matches.iloc[0]
    match_deliveries = deliveries_df[deliveries_df.get("match_id") == match_id].copy()
    if innings_filter:
        match_deliveries = match_deliveries[match_deliveries["innings"].isin(innings_filter)]

    if match_deliveries.empty:
        raise ValueError("No deliveries found for match/innings selection")

    context = {
        "meta": {"match_id": match_id},
        "info": {
            "teams": [match_row.get("team_a", "Team A"), match_row.get("team_b", "Team B")],
            "dates": [match_row.get("date")],
            "venue": match_row.get("venue"),
            "outcome": {"winner": match_row.get("winner")},
        },
    }

    deliveries_records: List[Dict[str, Any]] = match_deliveries.to_dict("records")  # type: ignore
    phase_overall = compute_phase_run_rates(deliveries_records, context)
    phase_team = compute_phase_run_rates_per_team(deliveries_records, context)
    wicket_clusters_df = compute_wicket_clusters_sliding(match_deliveries)
    wicket_clusters: List[Dict[str, object]] = [
        {
            "innings": int(rec.get("innings", 0)),
            "start_over": int(rec.get("start_over", 0)),
            "end_over": int(rec.get("end_over", 0)),
            "wickets": int(rec.get("wickets", 0)),
        }
        for rec in wicket_clusters_df.to_dict("records")
    ]
    bowler_clusters = compute_per_bowler_wicket_clusters(deliveries_records, context)
    monte_stats = monte_carlo_simulation(deliveries_records, context, trials=monte_trials)

    innings_summary = compute_innings_summary(match_deliveries)
    over_summary = compute_over_summary(match_deliveries)
    enriched_balls = enrich_ball_by_ball(match_deliveries, players_map)

    overview = {
        "match": {
            "match_id": match_id,
            "title": format_match_title(match_row),
            "team_a": match_row.get("team_a"),
            "team_b": match_row.get("team_b"),
            "venue": match_row.get("venue"),
            "date": match_row.get("date"),
            "winner": match_row.get("winner"),
        },
        "phase_overall": phase_overall,
        "phase_team": phase_team,
        "wicket_clusters": wicket_clusters,
        "bowler_clusters": bowler_clusters,
        "monte_carlo": monte_stats,
        "innings_summary": innings_summary.to_dict("records"),
        "over_summary": over_summary.to_dict("records"),
        "ball_by_ball": enriched_balls.head(400).to_dict("records"),
    }
    return overview


def format_match_title(row: pd.Series) -> str:
    if row is None or row.empty:
        return "Select a match"
    team_a = row.get("team_a", "Team A")
    team_b = row.get("team_b", "Team B")
    venue = row.get("venue", "")
    start_date = row.get("date", "")
    end_date = row.get("end_date", "")
    series_name = row.get("series_name")

    date_label = ""
    if start_date and end_date and str(end_date) != str(start_date):
        date_label = f"{start_date} to {end_date}"
    elif start_date:
        date_label = str(start_date)

    title = series_name if series_name else f"{team_a} vs {team_b}"
    if venue and date_label:
        return f"{title} — {venue} ({date_label})"
    if venue:
        return f"{title} — {venue}"
    if date_label:
        return f"{title} ({date_label})"
    return title
