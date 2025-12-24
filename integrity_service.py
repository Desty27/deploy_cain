from __future__ import annotations

from typing import Any, Dict, Optional
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


def analyze(match_id: int, overs_window: int = 3) -> Dict[str, Any]:
    deliveries_df, matches_df, _ = load_frames()
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
        return {
            "alerts": insights.alerts,
            "high_verdict_windows": insights.high_verdict_windows,
            "pressure_windows": pressure_records,
            "review_hotspots": review_records,
            "narrative": insights.narrative,
        }

    # Local fallback: compute basic pressure windows and alerts without global_scout
    df = match_deliveries.copy()
    df["runs_total"] = df["runs_off_bat"].fillna(0) + df["extras"].fillna(0)
    df["appeals"] = df["wicket_type"].notna().astype(int)
    df["lbw"] = df["wicket_type"].fillna("").str.contains("lbw", case=False).astype(int)
    df["runout"] = df["wicket_type"].fillna("").str.contains("run out", case=False).astype(int)

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

    return {
        "alerts": alerts,
        "high_verdict_windows": top_windows.to_dict("records") if not top_windows.empty else [],
        "pressure_windows": pressure_df.to_dict("records") if not pressure_df.empty else [],
        "review_hotspots": hotspots_rows,
        "narrative": None,
    }
