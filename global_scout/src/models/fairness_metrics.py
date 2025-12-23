from typing import Dict
import pandas as pd


def demographic_parity_diff(df: pd.DataFrame, group: str, decision: str) -> float:
    """Difference between max and min positive rates across groups."""
    if group not in df.columns or decision not in df.columns:
        return 0.0
    rates = df.groupby(group)[decision].mean().fillna(0.0)
    return float(rates.max() - rates.min()) if len(rates) > 0 else 0.0


def group_stats(df: pd.DataFrame, group: str, score_col: str) -> pd.DataFrame:
    if group not in df.columns or score_col not in df.columns:
        return pd.DataFrame(columns=[group, 'count', 'mean', 'std'])
    return df.groupby(group)[score_col].agg(['count', 'mean', 'std']).reset_index()


def equal_opportunity_diff(df: pd.DataFrame, group: str, decision: str, label_col: str) -> float:
    """True positive rate parity: P(decision=1 | label=1, group=g)."""
    if any(c not in df.columns for c in [group, decision, label_col]):
        return 0.0
    tprs: Dict[str, float] = {}
    for g, sub in df.groupby(group):
        pos = sub[sub[label_col] == 1]
        if len(pos) == 0:
            tprs[str(g)] = 0.0
        else:
            tprs[str(g)] = float(pos[decision].mean())
    return float(max(tprs.values()) - min(tprs.values())) if tprs else 0.0
