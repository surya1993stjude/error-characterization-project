import numpy as np
import pandas as pd

# --- Load CSV (use the file that already contains the *_rounded_second_decimal columns) ---
in_path = "C:\\Users\\ssarka62\\Documents\\GitHub\\error_characterization_project\\code and results\\3. code_for_tow_based_results\\computing_tow_scores\\computing_volume_tow\\output_file(s)\\Volume_and_surface_scores_with_C_scores_with_rounded_columns_and_tow_volume.csv"
out_path = "volume_E.csv"

df = pd.read_csv(in_path)

# --- Configure / auto-detect column names ---
# Try a few common variants; adjust if your CSV uses different headers.
def pick_col(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}\nAvailable columns: {list(df.columns)}")

volume_beta_col = pick_col([
    "beta_volume_rounded_second_decimal"
])
volume_dice_col = pick_col([
    "dice_volume_rounded_second_decimal"
])

# Ensure numeric (coerce strings to NaN if any)
df[volume_beta_col] = pd.to_numeric(df[volume_beta_col], errors="coerce")
df[volume_dice_col] = pd.to_numeric(df[volume_dice_col], errors="coerce")

# --- Compute volume area-based Error (E) for lambdas ---
lambdas = [0.00, 0.25, 0.50, 0.75, 1.00]

beta = df[volume_beta_col].to_numpy(dtype=float)
dsc  = df[volume_dice_col].to_numpy(dtype=float)

for lam in lambdas:
    E = np.sqrt((lam * beta) ** 2 + ((1.0 - lam) * (1.0 - dsc)) ** 2)
    df[f"volume_E_lambda_{lam:.2f}"] = E

# Save updated CSV
df.to_csv(out_path, index=False)

print("Done. Wrote:", out_path)
print("New columns added:", [c for c in df.columns if c.startswith("volume_E_lambda_")])
