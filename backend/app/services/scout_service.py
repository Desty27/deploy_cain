from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Ensure repo root (contains global_scout/) is importable when running from backend/
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

# Attempt to import the Global Scout pipelines; fall back gracefully if absent
try:
    from global_scout.src.pipelines.rank_candidates import score_candidates, mitigate_and_rank  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    score_candidates = None  # type: ignore
    mitigate_and_rank = None  # type: ignore


def _local_minmax(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    try:
        xv = float(x)
    except Exception:
        xv = lo
    xv = max(lo, min(hi, xv))
    return (xv - lo) / (hi - lo)


def _local_role_score(row: pd.Series) -> float:
    role = str(row.get("role", "")).lower()
    if role == "batter":
        sr = float(row.get("batting_sr") or 0.0)
        avg = float(row.get("batting_avg") or 0.0)
        form = float(row.get("recent_form") or 0.0)
        return 0.5 * _local_minmax(sr, 90, 180) + 0.4 * _local_minmax(avg, 15, 60) + 0.1 * form
    if role == "bowler":
        eco = float(row.get("bowling_eco") or 9.0)
        bavg = float(row.get("bowling_avg") or 45.0)
        form = float(row.get("recent_form") or 0.0)
        eco_s = 1 - _local_minmax(eco, 4.5, 10.0)
        bavg_s = 1 - _local_minmax(bavg, 15, 45)
        return 0.5 * eco_s + 0.4 * bavg_s + 0.1 * form
    sr = float(row.get("batting_sr") or 0.0)
    avg = float(row.get("batting_avg") or 0.0)
    eco = float(row.get("bowling_eco") or 9.0)
    bavg = float(row.get("bowling_avg") or 45.0)
    form = float(row.get("recent_form") or 0.0)
    eco_s = 1 - _local_minmax(eco, 4.5, 10.0)
    bavg_s = 1 - _local_minmax(bavg, 15, 45)
    bat = 0.5 * _local_minmax(sr, 90, 170) + 0.5 * _local_minmax(avg, 15, 55)
    bowl = 0.5 * eco_s + 0.5 * bavg_s
    return 0.45 * bat + 0.45 * bowl + 0.10 * form


def _local_strategic_fit(row: pd.Series) -> float:
    matches = float(row.get("matches") or 0.0)
    lvl = int(row.get("league_level") or 5)
    form = float(row.get("recent_form") or 0.0)
    exp_s = _local_minmax(matches, 5, 60)
    lvl_s = 1 - _local_minmax(lvl, 1, 5)
    return 0.5 * exp_s + 0.5 * form + 0.2 * lvl_s


def _local_score_candidates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["performance_score"] = df.apply(_local_role_score, axis=1)
    df["fit_score"] = df.apply(_local_strategic_fit, axis=1)
    df["raw_score"] = 0.75 * df["performance_score"] + 0.25 * df["fit_score"]
    return df


def _local_within_group_standardize(df: pd.DataFrame, group: str, score_col: str, out_col: str) -> pd.DataFrame:
    def zscore(s: pd.Series) -> pd.Series:
        mu, sd = s.mean(), s.std(ddof=0)
        return (s - mu) / (sd if sd and sd > 1e-8 else 1.0)

    df = df.copy()
    if group in df.columns and score_col in df.columns:
        df[out_col] = df.groupby(group)[score_col].transform(zscore)
    else:
        df[out_col] = df.get(score_col, 0.0)
    return df


def _local_monotonic_calibration(df: pd.DataFrame, score_col: str, out_col: str) -> pd.DataFrame:
    df = df.copy()
    x = df.get(score_col)
    if x is None:
        df[out_col] = 0.0
    else:
        df[out_col] = 1.0 / (1.0 + np.exp(-x.clip(-6, 6)))
    return df


def _local_group_stats(df: pd.DataFrame, group: str, score_col: str) -> pd.DataFrame:
    if group not in df.columns or score_col not in df.columns:
        return pd.DataFrame(columns=[group, "count", "mean", "std"])
    return df.groupby(group)[score_col].agg(["count", "mean", "std"]).reset_index()


def _local_demographic_parity_diff(df: pd.DataFrame, group: str, decision: str) -> float:
    if group not in df.columns or decision not in df.columns:
        return 0.0
    rates = df.groupby(group)[decision].mean().fillna(0.0)
    return float(rates.max() - rates.min()) if len(rates) > 0 else 0.0


def _local_equal_opportunity_diff(df: pd.DataFrame, group: str, decision: str, label_col: str) -> float:
    if any(c not in df.columns for c in [group, decision, label_col]):
        return 0.0
    tprs = []
    for _, sub in df.groupby(group):
        pos = sub[sub[label_col] == 1]
        tprs.append(float(pos[decision].mean()) if len(pos) else 0.0)
    return float(max(tprs) - min(tprs)) if tprs else 0.0


def _local_rank_candidates(df: pd.DataFrame, protected: str = "region", shortlist_k: int = 10) -> tuple[pd.DataFrame, Dict[str, Any]]:
    df = _local_score_candidates(df)
    df = _local_within_group_standardize(df, protected, "raw_score", "std_score")
    df = _local_monotonic_calibration(df, "std_score", "final_score")
    ranked = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    cutoff = ranked["final_score"].quantile(1 - min(1, shortlist_k / max(1, len(ranked))))
    ranked["shortlisted"] = (ranked["final_score"] >= cutoff).astype(int)
    baseline_cutoff = ranked["raw_score"].quantile(1 - min(1, shortlist_k / max(1, len(ranked))))
    ranked["baseline_label"] = (ranked["raw_score"] >= baseline_cutoff).astype(int)
    audits = {
        "group_stats": _local_group_stats(ranked, protected, "final_score").to_dict(orient="records"),
        "demographic_parity_diff": _local_demographic_parity_diff(ranked, protected, "shortlisted"),
        "equal_opportunity_diff": _local_equal_opportunity_diff(ranked, protected, "shortlisted", "baseline_label"),
        "cutoff": float(cutoff),
        "baseline_cutoff": float(baseline_cutoff),
    }
    return ranked, audits


def _build_response(
    input_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    audits: Dict[str, Any],
    protected: str,
    shortlist_k: int,
) -> Dict[str, Any]:
    input_preview = input_df.head(20).to_dict("records")

    fairness_heatmap = []
    heatmap_df = pd.DataFrame(audits.get("group_stats", []))
    if not heatmap_df.empty and protected in heatmap_df.columns:
        shortlist_rates = (
            ranked_df.groupby(protected)["shortlisted"].mean().reset_index(name="shortlist_rate")
            if protected in ranked_df.columns and "shortlisted" in ranked_df.columns
            else pd.DataFrame(columns=[protected, "shortlist_rate"])
        )
        fairness_heatmap = (
            heatmap_df.merge(shortlist_rates, on=protected, how="left")
            .rename(columns={"mean": "avg_final_score"})
            .fillna({"shortlist_rate": 0.0})
            .to_dict(orient="records")
        )

    top_shortlist = ranked_df[ranked_df.get("shortlisted", 0) == 1] if "shortlisted" in ranked_df.columns else ranked_df
    top_roles: Dict[str, Any] = {}
    if "role" in ranked_df.columns:
        for role in sorted(ranked_df["role"].dropna().astype(str).unique()):
            top_roles[role] = top_shortlist[top_shortlist["role"] == role].head(5).to_dict("records")

    top_groups: Dict[str, Any] = {}
    if protected in ranked_df.columns:
        for group_val in sorted(ranked_df[protected].dropna().astype(str).unique()):
            top_groups[group_val] = top_shortlist[top_shortlist[protected] == group_val].head(5).to_dict("records")

    return {
        "ranked": ranked_df.to_dict("records"),
        "audits": audits,
        "input_preview": input_preview,
        "fairness_heatmap": fairness_heatmap,
        "top_roles": top_roles,
        "top_groups": top_groups,
        "protected": protected,
        "shortlist_k": shortlist_k,
    }


class ScoutUnavailable(RuntimeError):
    pass


def _ensure_available() -> None:
    if score_candidates is None or mitigate_and_rank is None:
        return


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
    input_df = df.copy()
    if score_candidates is not None and mitigate_and_rank is not None:
        scored = score_candidates(df)  # type: ignore[misc]
        ranked_df, audits = mitigate_and_rank(scored, protected=protected, shortlist_k=shortlist_k)  # type: ignore[misc]
        ranked_df = ranked_df if isinstance(ranked_df, pd.DataFrame) else pd.DataFrame(ranked_df)
        return _build_response(input_df, ranked_df, audits, protected, shortlist_k)

    ranked_df, audits = _local_rank_candidates(df=input_df, protected=protected, shortlist_k=shortlist_k)
    return _build_response(input_df, ranked_df, audits, protected, shortlist_k)


def rank_from_payload(payload: Dict[str, Any], protected: str, shortlist_k: int) -> Dict[str, Any]:
    df = pd.DataFrame(payload)
    if df.empty:
        raise ValueError("No candidate rows provided")
    return rank_candidates(df=df, protected=protected, shortlist_k=shortlist_k)


def rank_demo(protected: str, shortlist_k: int) -> Dict[str, Any]:
    df = load_demo_candidates()
    return rank_candidates(df=df, protected=protected, shortlist_k=shortlist_k)
