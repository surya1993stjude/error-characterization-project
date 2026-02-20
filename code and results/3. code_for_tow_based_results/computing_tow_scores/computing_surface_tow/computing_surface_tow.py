import numpy as np
import pandas as pd

# --- Load CSV (use the file that already contains the *_rounded_second_decimal columns) ---
in_csv = "..\\round_off_columns_in_results_csv\\output_file(s)\\Volume_and_surface_scores_with_C_scores_with_rounded_columns.csv"
df = pd.read_csv(in_csv)

# --- Use ONLY the rounded columns ---
BETA_COL = "beta_surface_rounded_second_decimal"
DSC_COL  = "surface_dice_1p0mm_rounded_second_decimal"

# Sanity check (fails fast if you pointed at the wrong CSV)
missing = [c for c in [BETA_COL, DSC_COL] if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing columns {missing}. Make sure you're using the CSV that already has rounded columns."
    )

beta = df[BETA_COL].astype(float)
dsc  = df[DSC_COL].astype(float)

# --- Combined score (tau / tow) definition ---
# if beta == 0:  tau = λ + (1-λ)*DSC
# if beta != 0:  tau = λ * log10(1/|beta|)/2.1 + (1-λ)*DSC
def combined_surface_score(beta_series: pd.Series, dsc_series: pd.Series, lam: float) -> pd.Series:
    beta_abs = beta_series.abs()

    # log10(1/|beta|) is undefined for beta==0; compute only where beta!=0
    log_term = np.where(
        beta_abs.eq(0) | beta_abs.isna(),
        np.nan,
        np.log10(1.0 / beta_abs.to_numpy())
    )

    tau_when_beta_nonzero = lam * (log_term / 2.1) + (1.0 - lam) * dsc_series.to_numpy()
    tau_when_beta_zero    = lam + (1.0 - lam) * dsc_series.to_numpy()

    tau = np.where(beta_abs.eq(0).to_numpy(), tau_when_beta_zero, tau_when_beta_nonzero)

    # If beta or dsc is NaN, you may prefer tau to be NaN as well:
    tau = np.where(beta_series.isna().to_numpy() | dsc_series.isna().to_numpy(), np.nan, tau)

    return pd.Series(tau, index=beta_series.index)

# --- Create columns for the requested lambdas ---
lambdas = [0.00, 0.25, 0.50, 0.75, 1.00]
for lam in lambdas:
    # column name (you can rename this if you prefer a different convention)
    col_name = f"tow_surface_lambda_{lam:.2f}"
    df[col_name] = combined_surface_score(beta, dsc, lam)

# --- Save result ---
out_csv = "Volume_and_surface_scores_with_C_scores_with_rounded_columns_and_tow_surface.csv"
df.to_csv(out_csv, index=False)

print("Wrote:", out_csv)
print("Added columns:", [f"tow_surface_lambda_{lam:.2f}" for lam in lambdas])
