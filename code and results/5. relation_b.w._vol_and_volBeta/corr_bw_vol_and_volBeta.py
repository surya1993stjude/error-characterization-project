# import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV
in_path = "C:\\Users\\ssarka62\\Documents\\GitHub\\error_characterization_project\\code and results\\3. code_for_tow_based_results\\computing_tow_scores\\computing_volume_tow\\output_file(s)\\Volume_and_surface_scores_with_C_scores_with_rounded_columns_and_tow_volume.csv"
df = pd.read_csv(in_path)

# Select relevant columns and drop missing values
data = df[['beta_volume', 'gt_volume_mm3']].dropna()

# Extract arrays
beta_volume = data['beta_volume']
gt_volume = data['gt_volume_mm3']

# Pearson correlation (linear correlation)
pearson_corr, pearson_p = pearsonr(beta_volume, gt_volume)

# Spearman correlation (rank correlation)
spearman_corr, spearman_p = spearmanr(beta_volume, gt_volume)

print(f"Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4e})")
print(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4e})")



# ------------------------------------------------------------------------------


sns.scatterplot(x="gt_volume_mm3", y="beta_volume", data=df)
plt.show()


# ------------------------------------------------------------------------------


import matplotlib.pyplot as plt

plt.scatter(df['gt_volume_mm3'], df['beta_volume'], alpha=0.6)
plt.xlabel("Ground Truth Volume (mm³)")
plt.ylabel("Beta Volume Score")
plt.title("Beta Volume Score vs Ground Truth Volume")
plt.show()



# ============================================================================================================


results = []

for organ, subset in df.groupby("organ"):
    pearson_r, pearson_p = pearsonr(subset["beta_volume"], subset["gt_volume_mm3"])
    spearman_r, spearman_p = spearmanr(subset["beta_volume"], subset["gt_volume_mm3"])

    results.append({
        "organ": organ,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p
    })

results_df = pd.DataFrame(results)
print(results_df)






import seaborn as sns
import matplotlib.pyplot as plt

sns.lmplot(
    data=df,
    x="gt_volume_mm3",
    y="beta_volume",
    col="organ",
    scatter_kws={"alpha":0.6},
    height=4,
    aspect=1
)

plt.show()



# ==========================================================================================================================


import pandas as pd
from scipy.stats import spearmanr
import numpy as np

#  = pd.read_csv("Volume_and_surface_scores_with_C_scores_with_rounded_columns_and_tow_volume.csv")

results = []

for organ, g in df.groupby("organ"):
    
    # correlation between beta score and organ size
    r, p = spearmanr(g["beta_volume"], g["gt_volume_mm3"])
    
    # median organ size
    median_volume = g["gt_volume_mm3"].median()
    
    results.append({
        "organ": organ,
        "median_volume": median_volume,
        "correlation": r,
        "p_value": p
    })

corr_df = pd.DataFrame(results)
print(corr_df)


import matplotlib.pyplot as plt

plt.figure(figsize=(6,5))

plt.scatter(
    corr_df["median_volume"],
    corr_df["correlation"],
    s=120
)

for _, row in corr_df.iterrows():
    label = f"{row['organ']}\np={row['p_value']:.3f}"
    
    plt.text(
        row["median_volume"],
        row["correlation"],
        label,
        ha="center",
        va="bottom"
    )

plt.axhline(0, linestyle="--")

plt.xlabel("Median organ volume (mm³)")
plt.ylabel("Spearman correlation\n(beta_volume vs gt_volume)")
plt.title("Size bias of beta volume score per organ")

plt.tight_layout()
plt.show()





