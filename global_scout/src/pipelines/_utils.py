def minmax(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    try:
        xv = float(x)
    except Exception:
        xv = lo
    xv = max(lo, min(hi, xv))
    return (xv - lo) / (hi - lo)
