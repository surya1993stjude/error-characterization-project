import pandas as pd
import numpy as np
from pathlib import Path


# --- Paths ---
in_path = Path("Volume and surface scores for selected organs - test_with_organ_zscores.csv")
out_path = in_path.with_name(in_path.stem + "_and_with_overall_non_organ_specific_z_scores.csv")

# --- Columns to z-score ---
cols = [
    "dice_volume",
    "surface_dice_1p0mm",
    "beta_volume",
    "beta_surface",
]

# Choose standard deviation definition:
# ddof=0 -> population std (common for z-scores)
# ddof=1 -> sample std
DDOF = 0

# --- Load ---
df = pd.read_csv(in_path)

# --- Compute z-scores ---
for c in cols:
    if c not in df.columns:
        raise KeyError(f"Missing required column: {c}")

    s = (
        pd.to_numeric(df[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
              )    
    # s = pd.to_numeric(df[c], errors="coerce")  # non-numeric -> NaN
    mean = s.mean(skipna=True)
    std = s.std(skipna=True, ddof=DDOF)

    z_col = f"{c}_overall_z"
    if pd.isna(std) or std == 0:
        # If std is 0 (or all NaN), z-score isn't defined; keep as NaN
        df[z_col] = pd.NA
    else:
        df[z_col] = (s - mean) / std

# --- Save ---
df.to_csv(out_path, index=False)
print(f"Saved: {out_path}")