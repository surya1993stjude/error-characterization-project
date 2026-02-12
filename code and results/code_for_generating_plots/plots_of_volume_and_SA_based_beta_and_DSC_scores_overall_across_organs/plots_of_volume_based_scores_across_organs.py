import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# read csv file with scores
# ---------------------------
CSV_IN  = Path("Volume and surface scores for selected organs - 564_with_organ_zscores_and_with_overall_non_organ_specific_z_scores.csv")


# ---------------------------
# Load CSV
# ---------------------------
df = pd.read_csv(CSV_IN)


# ==================================================================================

# ================
# Generate plots:
# ================


organs_list = list(set(df["organ"]))

df_=df.copy()

df_long = df_.melt(
        id_vars="organ",                          # keep this as x-axis category
        value_vars=["dice_volume", "dice_volume_z", "beta_volume", "beta_volume_z"],  # scores become hue groups
        # value_vars=["surface_dice_1p0mm", "surface_dice_1p0mm_overall_z", "beta_surface", "beta_surface_overall_z"],  # scores become hue groups
        var_name="Score type",
        value_name="Score value"
    )
    
df_long.head()
    
EXCLUDE = "ribs"   # the category you want to remove
df_plot = df_long[df_long["organ"] != EXCLUDE]
EXCLUDE = "glands"
df_plot = df_plot[df_plot["organ"] != EXCLUDE]
    
# order=["dice_volume", "dice_volume_z", "beta_volume", "beta_volume_z"]



sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
    
sns.boxplot(
        data=df_plot,
        x="organ",        # your categorical column
        y="Score value",        # numeric values
        hue="Score type",  # Score1..Score4 grouped within each Group
        width=0.7,
        showfliers=False
        # order=order
    )
    
plt.title("Volume-based scores across organs")
    
# Short labels for x-axis ticks
short_labels = {
        "liver": "Liver",
        "stomach": "Stomach",
        "pancreas": "Pancreas"
    }
    
ax = plt.gca()
    
ax.set_xticklabels(
        [short_labels.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()]
    )
    
    
plt.tight_layout()
plt.savefig("volume_based_scores_overall_across_organs_without_outliers.png", dpi=300, bbox_inches="tight")
plt.show()