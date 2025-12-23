from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from . import _utils
from ..models.fairness_metrics import demographic_parity_diff, equal_opportunity_diff, group_stats
from ..models.bias_mitigation import within_group_standardize, monotonic_calibration

DEFAULT_PROTECTED = "region"


def _role_score(row: pd.Series) -> float:
    role = str(row.get("role", "")).lower()
    # Role-aware weighting; missing metrics default safely
    if role == "batter":
        sr = float(row.get("batting_sr") or 0.0)
        avg = float(row.get("batting_avg") or 0.0)
        form = float(row.get("recent_form") or 0.0)
        return 0.5 * _utils.minmax(sr, 90, 180) + 0.4 * _utils.minmax(avg, 15, 60) + 0.1 * form
    if role == "bowler":
        eco = float(row.get("bowling_eco") or 9.0)
        bavg = float(row.get("bowling_avg") or 45.0)
        form = float(row.get("recent_form") or 0.0)
        # Lower is better -> invert
        eco_s = 1 - _utils.minmax(eco, 4.5, 10.0)
        bavg_s = 1 - _utils.minmax(bavg, 15, 45)
        return 0.5 * eco_s + 0.4 * bavg_s + 0.1 * form
    # allrounder or keeper: blend
    sr = float(row.get("batting_sr") or 0.0)
    avg = float(row.get("batting_avg") or 0.0)
    eco = float(row.get("bowling_eco") or 9.0)
    bavg = float(row.get("bowling_avg") or 45.0)
    form = float(row.get("recent_form") or 0.0)
    eco_s = 1 - _utils.minmax(eco, 4.5, 10.0)
    bavg_s = 1 - _utils.minmax(bavg, 15, 45)
    bat = 0.5 * _utils.minmax(sr, 90, 170) + 0.5 * _utils.minmax(avg, 15, 55)
    bowl = 0.5 * eco_s + 0.5 * bavg_s
    return 0.45 * bat + 0.45 * bowl + 0.10 * form


def strategic_fit(row: pd.Series) -> float:
    # Penalize very low matches, reward higher league levels and recent form
    matches = float(row.get("matches") or 0.0)
    lvl = int(row.get("league_level") or 5)
    form = float(row.get("recent_form") or 0.0)
    exp_s = _utils.minmax(matches, 5, 60)
    lvl_s = 1 - _utils.minmax(lvl, 1, 5)  # elite=1 -> higher score
    return 0.5 * exp_s + 0.5 * form + 0.2 * lvl_s


def score_candidates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["performance_score"] = df.apply(_role_score, axis=1)
    df["fit_score"] = df.apply(strategic_fit, axis=1)
    # Blend with small regularization towards fit to reduce stat-padding bias
    df["raw_score"] = 0.75 * df["performance_score"] + 0.25 * df["fit_score"]
    return df


def mitigate_and_rank(df: pd.DataFrame, protected: str = DEFAULT_PROTECTED, shortlist_k: int = 10) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # Bias mitigation by within-group standardization and global calibration
    df = within_group_standardize(df, protected, "raw_score", "std_score")
    df = monotonic_calibration(df, "std_score", "final_score")
    ranked = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)

    # Audits at a reasonable shortlist threshold (decision after mitigation)
    cutoff = ranked["final_score"].quantile(1 - min(1, shortlist_k / max(1, len(ranked))))
    ranked["shortlisted"] = (ranked["final_score"] >= cutoff).astype(int)

    # Independent baseline label from pre-mitigation raw_score (proxy for "true positive" potential)
    baseline_cutoff = ranked["raw_score"].quantile(1 - min(1, shortlist_k / max(1, len(ranked))))
    ranked["baseline_label"] = (ranked["raw_score"] >= baseline_cutoff).astype(int)

    audits = {
        "group_stats": group_stats(ranked, protected, "final_score").to_dict(orient="records"),
        "demographic_parity_diff": demographic_parity_diff(ranked, protected, "shortlisted"),
        # EO measures TPR parity: P(decision=1 | label=1, group)
        # Use baseline_label as the label to avoid tautology (label != decision)
        "equal_opportunity_diff": equal_opportunity_diff(ranked, protected, "shortlisted", "baseline_label"),
        "cutoff": float(cutoff),
        "baseline_cutoff": float(baseline_cutoff),
    }
    return ranked, audits


def run_pipeline(input_csv: Path, output_csv: Path, protected: str = DEFAULT_PROTECTED, shortlist_k: int = 10) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = pd.read_csv(input_csv)
    df = score_candidates(df)
    ranked, audits = mitigate_and_rank(df, protected=protected, shortlist_k=shortlist_k)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output_csv, index=False)
    return ranked, audits
