import pandas as pd
import numpy as np


def within_group_standardize(df: pd.DataFrame, group: str, score_col: str, out_col: str) -> pd.DataFrame:
    """Z-score within each group to reduce distributional skews."""
    def zscore(s: pd.Series) -> pd.Series:
        mu, sd = s.mean(), s.std(ddof=0)
        return (s - mu) / (sd if sd and sd > 1e-8 else 1.0)

    df = df.copy()
    if group in df.columns and score_col in df.columns:
        df[out_col] = df.groupby(group)[score_col].transform(zscore)
    else:
        df[out_col] = df.get(score_col, 0.0)
    return df


def monotonic_calibration(df: pd.DataFrame, score_col: str, out_col: str) -> pd.DataFrame:
    """Map to 0..1 via sigmoid to aid interpretability and thresholds."""
    df = df.copy()
    x = df.get(score_col)
    if x is None:
        df[out_col] = 0.0
    else:
        df[out_col] = 1.0 / (1.0 + np.exp(-x.clip(-6, 6)))
    return df
