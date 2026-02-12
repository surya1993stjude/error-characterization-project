import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------
# Inputs
# ---------------------------
CSV_IN  = Path("Volume and surface scores for selected organs - test.csv")
CSV_OUT = Path("Volume and surface scores for selected organs - test_with_organ_zscores.csv")

# Columns to z-score (edit if needed)
Z_COLS = [
    "dice_volume",
    "surface_dice_1p0mm",   # adjust if your column name differs
    "beta_volume",
    "beta_surface",
]

# ---------------------------
# Load CSV
# ---------------------------
df = pd.read_csv(CSV_IN)

# ---------------------------
# Organ-wise z-score function
# ---------------------------
def zscore_series(x: pd.Series) -> pd.Series:
    """
    Population z-score: (x - mean) / std, computed within one organ.
    Returns NaN if std == 0 or all values are NaN.
    """
    # mu = x.mean(skipna=True)
    # sigma = x.std(skipna=True, ddof=0)
    
    s_clean = (
            pd.to_numeric(x, errors="coerce")
              .replace([np.inf, -np.inf], np.nan)
              .dropna()
              )
    mu = s_clean.mean(skipna=True)
    sigma = s_clean.std(ddof=0)   # pandas std
    print(sigma)
    
    # if sigma == 0 or np.isnan(sigma):
    #     return pd.Series(np.nan, index=x.index)
    print((s_clean - mu) / sigma)
    return (s_clean - mu) / sigma

# ---------------------------
# Compute z-scores per organ
# ---------------------------
for col in Z_COLS:
    if col not in df.columns:
        print(f"Skipping missing column: {col}")
        continue

    df[f"{col}_z"] = (
        df
        .groupby("organ", group_keys=False)[col]
        .transform(zscore_series)
    )

# ---------------------------
# Save
# ---------------------------
df.to_csv(CSV_OUT, index=False)
print(f"Saved organ-wise z-scores to:\n{CSV_OUT.resolve()}")
