import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# read csv file with scores
# ---------------------------
CSV_IN  = Path("Volume and surface scores for selected organs - test_with_organ_zscores.csv")


# ---------------------------
# Load CSV
# ---------------------------
df = pd.read_csv(CSV_IN)


# ==================================================================================

# ================
# Generate plots:
# ================


organs_list = list(set(df["organ"]))
for organ_name in organs_list:
    df_=df[df["organ"]==organ_name]

    df_long = df_.melt(
        id_vars="interpretation_surface",                          # keep this as x-axis category
        # value_vars=["dice_volume","dice_volume_z","surface_dice_1p0mm","surface_dice_1p0mm_z", "beta_volume", "beta_volume_z", "beta_surface", "beta_surface_z"],  # scores become hue groups
        value_vars=["surface_dice_1p0mm", "surface_dice_1p0mm_z", "beta_surface", "beta_surface_z"],  # scores become hue groups
        var_name="Score type",
        value_name="Score value"
    )
    
    df_long.head()
    
    EXCLUDE = "GT empty, prediction non-empty (beta=inf): false positive / over-seg"   # the category you want to remove
    df_plot = df_long[df_long["interpretation_surface"] != EXCLUDE]
    EXCLUDE = "surface dice undefined (both surfaces empty)"
    df_plot = df_plot[df_plot["interpretation_surface"] != EXCLUDE]
    
    # order=["dice_volume", "dice_volume_z", "beta_volume", "beta_volume_z"]



    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    sns.boxplot(
        data=df_plot,
        x="interpretation_surface",        # your categorical column
        y="Score value",        # numeric values
        hue="Score type",  # Score1..Score4 grouped within each Group
        width=0.7,
        showfliers=False
        # order=order
    )
    
    plt.title(f"SA-based scores for {organ_name}")
    
    # Short labels for x-axis ticks
    short_labels = {
        # "GT empty, prediction non-empty (beta=inf): false positive / over-seg": "beta>0:\nover-segmentation",
        "beta≈0 & high DSC: (near) perfect segmentation": "beta≈0 & high DSC:\n(near) perfect segmentation",
        "beta≈0 & low DSC: size matches but poor overlap (mis-segmentation)": "beta≈0 & low DSC:\nsize matches but\npoor overlap\n(mis-segmentation)",
        "beta>0: over-segmentation": "beta>0:\nover-segmentation",
        "beta<0: under-segmentation": "beta<0:\nunder-segmentation"
    }
    
    ax = plt.gca()
    
    ax.set_xticklabels(
        [short_labels.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()]
    )
    
    
    plt.tight_layout()
    plt.savefig(f"{organ_name}_SA_based_scores_without_outliers.png", dpi=300, bbox_inches="tight")
    plt.show()