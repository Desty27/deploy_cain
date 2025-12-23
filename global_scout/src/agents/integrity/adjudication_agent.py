from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from global_scout.src.services.azure_openai import generate_integrity_brief


@dataclass
class IntegrityInsights:
    match_id: int
    innings: List[int]
    high_verdict_windows: List[Dict[str, object]]
    review_hotspots: pd.DataFrame
    pressure_windows: pd.DataFrame
    alerts: List[str]
    narrative: Optional[str] = None


class AdjudicationIntegrityAgent:
    """Enrich umpiring support with anomaly detection and LLM briefs."""

    def __init__(self, deliveries: pd.DataFrame, match_row: pd.Series, *, overs_window: int = 3) -> None:
        if deliveries.empty:
            raise ValueError("Deliveries dataframe cannot be empty for adjudication analysis")
        self.deliveries = deliveries.copy()
        self.match_row = match_row
        self.match_id = int(match_row.get("match_id")) if "match_id" in match_row else -1
        self.teams = [match_row.get("team_a", "Team A"), match_row.get("team_b", "Team B")]
        self.overs_window = overs_window

    def _windowize(self, df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["innings", "start_over", "end_over"] + target_cols)
        rows: List[Dict[str, object]] = []
        for innings in sorted(df["innings"].dropna().astype(int).unique()):
            inn_df = df[df["innings"] == innings]
            max_over = int(inn_df["over"].max()) if not inn_df.empty else 0
            for start in range(0, max_over + 1, self.overs_window):
                end = min(start + self.overs_window - 1, max_over)
                window_df = inn_df[(inn_df["over"] >= start) & (inn_df["over"] <= end)]
                record: Dict[str, object] = {"innings": innings, "start_over": start, "end_over": end}
                for col in target_cols:
                    record[col] = float(window_df[col].sum()) if col in window_df else 0.0
                rows.append(record)
        return pd.DataFrame(rows)

    def _compute_pressure_windows(self) -> pd.DataFrame:
        df = self.deliveries.copy()
        df["runs_total"] = df["runs_off_bat"].fillna(0) + df["extras"].fillna(0)
        df["appeals"] = df["wicket_type"].notna().astype(int)
        df["lbw"] = df["wicket_type"].fillna("").str.contains("lbw", case=False).astype(int)
        df["runout"] = df["wicket_type"].fillna("").str.contains("run out", case=False).astype(int)
        window_df = self._windowize(df, ["runs_total", "appeals", "lbw", "runout"])
        if window_df.empty:
            return window_df
        window_df["pressure_index"] = (
            window_df["appeals"] * 1.5 + window_df["lbw"] * 2.0 + window_df["runout"] * 1.0
        ) / self.overs_window
        window_df["scoring_rate"] = window_df["runs_total"] / np.maximum(self.overs_window * 6, 1)
        return window_df.sort_values(by="pressure_index", ascending=False)

    def _compute_review_hotspots(self, pressure_df: pd.DataFrame) -> pd.DataFrame:
        if pressure_df.empty:
            return pd.DataFrame(columns=["innings", "over", "pressure_index", "reason"])
        top_windows = pressure_df.head(5)
        rows: List[Dict[str, object]] = []
        for _, window in top_windows.iterrows():
            innings = int(window["innings"])
            mask = (
                (self.deliveries["innings"].astype(int) == innings)
                & (self.deliveries["over"].astype(int) >= int(window["start_over"]))
                & (self.deliveries["over"].astype(int) <= int(window["end_over"]))
            )
            clip = self.deliveries[mask].copy()
            clip["lbw"] = clip["wicket_type"].fillna("").str.contains("lbw", case=False).astype(int)
            clip["runout"] = clip["wicket_type"].fillna("").str.contains("run out", case=False).astype(int)
            rows.append(
                {
                    "innings": innings,
                    "over": f"{int(window['start_over'])}-{int(window['end_over'])}",
                    "pressure_index": float(window["pressure_index"]),
                    "lbw_calls": int(clip["lbw"].sum()),
                    "runout_calls": int(clip["runout"].sum()),
                    "appeals": int(window["appeals"]),
                    "reason": (
                        "LBW concentration" if clip["lbw"].sum() >= clip["runout"].sum() else "Run-out pressure"
                    ),
                }
            )
        return pd.DataFrame(rows)

    def _identify_alerts(self, pressure_df: pd.DataFrame) -> List[str]:
        alerts: List[str] = []
        if pressure_df.empty:
            alerts.append("Low appeal volume detected; integrity risk minimal this match.")
            return alerts
        emitted: set[str] = set()

        def enqueue(message: str, row: pd.Series) -> None:
            key = f"{int(row['innings'])}-{int(row['start_over'])}-{int(row['end_over'])}-{message.split(':')[0]}"
            if key in emitted:
                return
            emitted.add(key)
            alerts.append(message)

        lbw_spikes = pressure_df.sort_values("lbw", ascending=False)
        for _, row in lbw_spikes.iterrows():
            if row["lbw"] < 3:
                break
            enqueue(
                f"LBW review surge: Innings {int(row['innings'])}, overs {int(row['start_over'])}-{int(row['end_over'])} (lbw={int(row['lbw'])}).",
                row,
            )
            if len(emitted) >= 3:
                break

        runout_flurry = pressure_df.sort_values("runout", ascending=False)
        for _, row in runout_flurry.iterrows():
            if row["runout"] < 2:
                break
            enqueue(
                f"Run-out hotspot: Innings {int(row['innings'])}, overs {int(row['start_over'])}-{int(row['end_over'])} (run-outs={int(row['runout'])}).",
                row,
            )
            if len(emitted) >= 6:
                break

        appeals_heavy = pressure_df.sort_values("appeals", ascending=False)
        for _, row in appeals_heavy.iterrows():
            if row["appeals"] < 4:
                break
            enqueue(
                f"Appeal stress window: Innings {int(row['innings'])}, overs {int(row['start_over'])}-{int(row['end_over'])} with {int(row['appeals'])} appeals.",
                row,
            )
            if len(emitted) >= 9:
                break
        if not alerts:
            alerts.append("No adjudication anomalies detected; maintain standard oversight.")
        return alerts

    def build_insights(self) -> IntegrityInsights:
        pressure_windows = self._compute_pressure_windows()
        hotspots = self._compute_review_hotspots(pressure_windows)
        alerts = self._identify_alerts(pressure_windows)

        narrative = generate_integrity_brief(
            match_meta={
                "match_id": self.match_id,
                "teams": self.teams,
                "venue": self.match_row.get("venue", "Unknown venue"),
                "date": self.match_row.get("date", "TBD"),
            },
            pressure_windows=json.loads(pressure_windows.head(6).to_json(orient="records")),
            hotspots=json.loads(hotspots.to_json(orient="records")) if not hotspots.empty else [],
            alerts=alerts,
        )

        return IntegrityInsights(
            match_id=self.match_id,
            innings=sorted(self.deliveries["innings"].dropna().astype(int).unique()),
            high_verdict_windows=json.loads(pressure_windows.head(6).to_json(orient="records")) if not pressure_windows.empty else [],
            review_hotspots=hotspots,
            pressure_windows=pressure_windows,
            alerts=alerts,
            narrative=narrative,
        )


def build_agent(deliveries: pd.DataFrame, match_row: pd.Series, overs_window: int = 3) -> AdjudicationIntegrityAgent:
    return AdjudicationIntegrityAgent(deliveries=deliveries, match_row=match_row, overs_window=overs_window)
