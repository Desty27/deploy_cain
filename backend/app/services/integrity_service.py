from __future__ import annotations

from typing import Any, Dict, Optional, List
from pathlib import Path
import sys

import pandas as pd

from app.services.data_loader import load_frames

# Ensure repo root (contains global_scout/) is importable when running from backend/
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

try:
    from global_scout.src.agents.integrity.adjudication_agent import build_agent  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    build_agent = None  # type: ignore


class IntegrityUnavailable(RuntimeError):
    pass


def _clean_wicket_type(value: Any) -> str:
    text = str(value or "").strip()
    if text in {"", "-", "None", "none", "nan", "NaN"}:
        return ""
    return text


def _compute_appeal_metrics(
    df: pd.DataFrame,
    players_map: Dict[int, str],
    overs_window: int,
    pressure_windows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    working = df.copy()
    working["wicket_type"] = working["wicket_type"].apply(_clean_wicket_type)
    working["innings"] = working["innings"].fillna(0).astype(int)
    working["over"] = working["over"].fillna(0).astype(int)
    working["ball"] = working["ball"].fillna(0).astype(int)
    working["appeals"] = working["wicket_type"].str.strip().ne("").astype(int)
    working["lbw"] = working["wicket_type"].str.contains("lbw", case=False).astype(int)
    working["runout"] = working["wicket_type"].str.contains("run out", case=False).astype(int)

    total_appeals = int(working["appeals"].sum())
    lbw_total = int(working["lbw"].sum())
    runout_total = int(working["runout"].sum())
    other_appeals = max(total_appeals - lbw_total - runout_total, 0)

    other_mix_rows = working[(working["appeals"] == 1) & (working["lbw"] == 0) & (working["runout"] == 0)].copy()
    other_mix_rows["wicket_label"] = (
        other_mix_rows["wicket_type"].astype(str).str.replace("_", " ", regex=False).str.title().replace({"": "Unspecified Appeal"})
    )
    other_mix = (
        other_mix_rows["wicket_label"].value_counts().reset_index().rename(columns={"index": "label", "wicket_label": "count"}).to_dict("records")
        if not other_mix_rows.empty
        else []
    )

    appeals_by_innings = (
        working.groupby("innings")["appeals"].sum().reset_index().rename(columns={"appeals": "count"}).assign(
            innings=lambda d: d["innings"].astype(int), count=lambda d: d["count"].astype(int)
        ).to_dict("records")
        if "innings" in working.columns
        else []
    )

    appeal_density = (
        working.groupby(["innings", "over"]).agg(
            appeals=("appeals", "sum"), lbw=("lbw", "sum"), runout=("runout", "sum")
        )
        .reset_index()
        .assign(
            innings=lambda d: d["innings"].astype(int),
            over=lambda d: d["over"].astype(int),
            appeals=lambda d: d["appeals"].astype(int),
            lbw=lambda d: d["lbw"].astype(int),
            runout=lambda d: d["runout"].astype(int),
        )
        .sort_values(["innings", "over"])
        .to_dict("records")
        if not working.empty
        else []
    )

    appeal_ledger = working[working["appeals"] == 1].copy()
    appeal_ledger["dismissed"] = appeal_ledger["dismissed_batter_id"].apply(
        lambda pid: players_map.get(int(pid), str(int(pid))) if not pd.isna(pid) else "-"
    )
    appeal_ledger["bowler"] = appeal_ledger["bowler_id"].apply(
        lambda pid: players_map.get(int(pid), str(int(pid))) if not pd.isna(pid) else "-"
    )
    appeal_ledger["batter"] = appeal_ledger["batter_id"].apply(
        lambda pid: players_map.get(int(pid), str(int(pid))) if not pd.isna(pid) else "-"
    )
    appeal_ledger_records = (
        appeal_ledger[
            ["innings", "over", "ball", "batter", "bowler", "wicket_type", "dismissed"]
        ]
        .assign(innings=lambda d: d["innings"].astype(int), over=lambda d: d["over"].astype(int), ball=lambda d: d["ball"].astype(int))
        .sort_values(["innings", "over", "ball"])
        .to_dict("records")
    )

    peak_pressure = 0.0
    for row in pressure_windows:
        try:
            peak_pressure = max(peak_pressure, float(row.get("pressure_index", 0)))
        except Exception:
            continue

    return {
        "total_appeals": total_appeals,
        "lbw_total": lbw_total,
        "runout_total": runout_total,
        "other_appeals": other_appeals,
        "other_appeals_mix": other_mix,
        "appeals_by_innings": appeals_by_innings,
        "appeal_density_by_over": appeal_density,
        "appeal_ledger": appeal_ledger_records,
        "peak_pressure_index": peak_pressure,
    }


def analyze(match_id: int, overs_window: int = 3) -> Dict[str, Any]:
    deliveries_df, matches_df, players_map = load_frames()
    match_row_df = matches_df[matches_df.get("match_id") == match_id]
    if match_row_df.empty:
        raise FileNotFoundError(f"match_id {match_id} not found")
    match_row = match_row_df.iloc[0]

    match_deliveries = deliveries_df[deliveries_df.get("match_id") == match_id].copy()
    if match_deliveries.empty:
        raise ValueError("No deliveries found for match")

    if build_agent is not None:
        agent = build_agent(match_deliveries, match_row, overs_window=overs_window)  # type: ignore[misc]
        insights = agent.build_insights()
        pressure_records = insights.pressure_windows.to_dict("records") if not insights.pressure_windows.empty else []
        review_records = insights.review_hotspots.to_dict("records") if not insights.review_hotspots.empty else []
        appeal_metrics = _compute_appeal_metrics(match_deliveries, players_map, overs_window, pressure_records)
        return {
            "alerts": insights.alerts,
            "high_verdict_windows": insights.high_verdict_windows,
            "pressure_windows": pressure_records,
            "review_hotspots": review_records,
            "narrative": insights.narrative,
            **appeal_metrics,
        }

    # Local fallback: compute basic pressure windows and alerts without global_scout
    df = match_deliveries.copy()
    df["wicket_type"] = df["wicket_type"].apply(_clean_wicket_type)
    df["runs_total"] = df["runs_off_bat"].fillna(0) + df["extras"].fillna(0)
    df["appeals"] = df["wicket_type"].str.strip().ne("").astype(int)
    df["lbw"] = df["wicket_type"].str.contains("lbw", case=False).astype(int)
    df["runout"] = df["wicket_type"].str.contains("run out", case=False).astype(int)

    rows = []
    for innings in sorted(df["innings"].dropna().astype(int).unique()):
        inn_df = df[df["innings"] == innings]
        max_over = int(inn_df["over"].max()) if not inn_df.empty else 0
        for start in range(0, max_over + 1, overs_window):
            end = min(start + overs_window - 1, max_over)
            clip = inn_df[(inn_df["over"] >= start) & (inn_df["over"] <= end)]
            rows.append(
                {
                    "innings": innings,
                    "start_over": start,
                    "end_over": end,
                    "runs_total": float(clip["runs_total"].sum()) if not clip.empty else 0.0,
                    "appeals": int(clip["appeals"].sum()) if not clip.empty else 0,
                    "lbw": int(clip["lbw"].sum()) if not clip.empty else 0,
                    "runout": int(clip["runout"].sum()) if not clip.empty else 0,
                }
            )

    pressure_df = pd.DataFrame(rows)
    if not pressure_df.empty:
        pressure_df["pressure_index"] = (
            pressure_df["appeals"] * 1.5 + pressure_df["lbw"] * 2.0 + pressure_df["runout"] * 1.0
        ) / max(1, overs_window)
        pressure_df["scoring_rate"] = pressure_df["runs_total"] / max(overs_window * 6, 1)
        pressure_df = pressure_df.sort_values(by="pressure_index", ascending=False)
    else:
        pressure_df = pd.DataFrame(columns=["innings", "start_over", "end_over", "pressure_index", "appeals", "lbw", "runout", "scoring_rate"])

    hotspots_rows = []
    top_windows = pressure_df.head(5)
    for _, window in top_windows.iterrows():
        innings = int(window["innings"])
        mask = (
            (df["innings"].astype(int) == innings)
            & (df["over"].astype(int) >= int(window["start_over"]))
            & (df["over"].astype(int) <= int(window["end_over"]))
        )
        clip = df[mask].copy()
        clip["lbw"] = clip["wicket_type"].fillna("").str.contains("lbw", case=False).astype(int)
        clip["runout"] = clip["wicket_type"].fillna("").str.contains("run out", case=False).astype(int)
        hotspots_rows.append(
            {
                "innings": innings,
                "over": f"{int(window['start_over'])}-{int(window['end_over'])}",
                "pressure_index": float(window.get("pressure_index", 0)),
                "lbw_calls": int(clip["lbw"].sum()) if not clip.empty else 0,
                "runout_calls": int(clip["runout"].sum()) if not clip.empty else 0,
                "appeals": int(window.get("appeals", 0)),
                "reason": "LBW concentration" if clip["lbw"].sum() >= clip["runout"].sum() else "Run-out pressure",
            }
        )

    alerts: list[str] = []
    if pressure_df.empty:
        alerts.append("Low appeal volume detected; integrity risk minimal this match.")
    else:
        for _, row in pressure_df.head(3).iterrows():
            if row.get("lbw", 0) >= 3:
                alerts.append(
                    f"LBW review surge: Innings {int(row['innings'])}, overs {int(row['start_over'])}-{int(row['end_over'])} (lbw={int(row['lbw'])})."
                )
        for _, row in pressure_df.head(3).iterrows():
            if row.get("runout", 0) >= 2:
                alerts.append(
                    f"Run-out hotspot: Innings {int(row['innings'])}, overs {int(row['start_over'])}-{int(row['end_over'])} (run-outs={int(row['runout'])})."
                )
        if not alerts:
            alerts.append("No adjudication anomalies detected; maintain standard oversight.")

    pressure_records = pressure_df.to_dict("records") if not pressure_df.empty else []
    appeal_metrics = _compute_appeal_metrics(match_deliveries, players_map, overs_window, pressure_records)

    return {
        "alerts": alerts,
        "high_verdict_windows": top_windows.to_dict("records") if not top_windows.empty else [],
        "pressure_windows": pressure_records,
        "review_hotspots": hotspots_rows,
        "narrative": None,
        **appeal_metrics,
    }
