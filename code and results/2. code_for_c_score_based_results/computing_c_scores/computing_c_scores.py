import pandas as pd

# -----------------------------
# Load input CSV
# -----------------------------
input_csv = "Volume and surface scores for selected organs - test_with_organ_zscores_and_with_overall_non_organ_specific_z_scores.csv"
df = pd.read_csv(input_csv)

# -----------------------------
# Parameters
# -----------------------------
lambdas = [0, 0.25, 0.5, 0.75, 1.0]

# Column names in your CSV
volume_beta_col = "beta_volume"
volume_dsc_col = "dice_volume"
surface_beta_col = "beta_surface"
surface_dsc_col = "surface_dice_1p0mm"

# -----------------------------
# Compute C scores
# -----------------------------
for lam in lambdas:
    df[f"C_volume_lambda_{lam}"] = (
        lam * ((df[volume_beta_col] + 1) / 2)
        + (1 - lam) * df[volume_dsc_col]
    )

    df[f"C_surface_lambda_{lam}"] = (
        lam * ((df[surface_beta_col] + 1) / 2)
        + (1 - lam) * df[surface_dsc_col]
    )

# -----------------------------
# Save output CSV
# -----------------------------
output_csv = "Volume_and_surface_scores_with_C_scores.csv"
df.to_csv(output_csv, index=False)

print(f"Saved updated CSV to: {output_csv}")