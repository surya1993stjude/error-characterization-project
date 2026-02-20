import numpy as np
import pandas as pd

# --- Load CSV (use the file that already contains the *_rounded_second_decimal columns) ---
in_csv = "C:\\Users\\ssarka62\\Documents\\GitHub\\error_characterization_project\\code and results\\1. code_for_beta_and_dsc_based_results\\round_off_columns_in_results_csv\\output_file(s)\\Volume_and_surface_scores_with_C_scores_with_rounded_columns.csv"
df = pd.read_csv(in_csv)

# --- Use ONLY the rounded columns (volume-based) ---
BETA_COL = "beta_volume_rounded_second_decimal"
DSC_COL  = "dice_volume_rounded_second_decimal"

missing = [c for c in [BETA_COL, DSC_COL] if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing columns {missing}. Make sure your CSV has the rounded columns for volume beta & volume DSC."
    )

beta = df[BETA_COL].astype(float)
dsc  = df[DSC_COL].astype(float)

# --- Combined score (tau / tow) definition ---
# if beta == 0:  tau = λ + (1-λ)*DSC
# if beta != 0:  tau = λ * log10(1/|beta|)/2.1 + (1-λ)*DSC
def combined_volume_score(beta_series: pd.Series, dsc_series: pd.Series, lam: float) -> pd.Series:
    beta_abs = beta_series.abs()

    log_term = np.where(
        beta_abs.eq(0) | beta_abs.isna(),
        np.nan,
        np.log10(1.0 / beta_abs.to_numpy())
    )

    tau_when_beta_nonzero = lam * (log_term / 2.1) + (1.0 - lam) * dsc_series.to_numpy()
    tau_when_beta_zero    = lam + (1.0 - lam) * dsc_series.to_numpy()

    tau = np.where(beta_abs.eq(0).to_numpy(), tau_when_beta_zero, tau_when_beta_nonzero)
    tau = np.where(beta_series.isna().to_numpy() | dsc_series.isna().to_numpy(), np.nan, tau)

    return pd.Series(tau, index=beta_series.index)

# --- Create columns for the requested lambdas ---
lambdas = [0.00, 0.25, 0.50, 0.75, 1.00]
for lam in lambdas:
    col_name = f"tow_volume_lambda_{lam:.2f}"
    df[col_name] = combined_volume_score(beta, dsc, lam)

# --- Save result ---
out_csv = "Volume_and_surface_scores_with_C_scores_with_rounded_columns_and_tow_volume.csv"
df.to_csv(out_csv, index=False)

print("Wrote:", out_csv)
print("Added columns:", [f"tow_volume_lambda_{lam:.2f}" for lam in lambdas])
