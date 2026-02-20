import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# read csv file with scores
# ---------------------------
CSV_IN  = Path("C:\\Users\\ssarka62\\Documents\\GitHub\\error_characterization_project\\code and results\\4. code_for_error_E_score\\computing_E_scores\\computing_volume_E\\output_file(s)\\volume_E.csv")


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
        id_vars="interpretation_volume",                          # keep this as x-axis category
        value_vars=['volume_E_lambda_0.00', 'volume_E_lambda_0.25', 'volume_E_lambda_0.50', 'volume_E_lambda_0.75', 'volume_E_lambda_1.00'],  # scores become hue groups
        var_name="Score type",
        value_name="Score value"
    )
    
    df_long.head()
    
    EXCLUDE = "GT empty, prediction non-empty (beta=inf): false positive / over-seg"   # the category you want to remove
    x_order = [
    "beta≈0 & high DSC: (near) perfect segmentation",
    "beta≈0 & low DSC: size matches but poor overlap (mis-segmentation)",
    "beta>0: over-segmentation",
    "beta<0: under-segmentation"]
    
    df_plot = df_long[df_long["interpretation_volume"] != EXCLUDE]
    
    counts = (
    df_plot.groupby("interpretation_volume")
    .size()
    .reindex(x_order)  # ensures correct order
    .fillna(0)
    .astype(int))
    
    counts=counts//5



    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    sns.boxplot(
        data=df_plot,
        x="interpretation_volume",        # your categorical column
        y="Score value",        # numeric values
        hue="Score type",  # Score1..Score4 grouped within each Group
        width=0.7,
        # showfliers=False,
        order=x_order
    )
    
    plt.title(f"Volume-based $Error$ $(E)$ for {organ_name}")
    
    # Short labels for x-axis ticks
    short_labels = {
        # "GT empty, prediction non-empty (beta=inf): false positive / over-seg": "beta>0:\nover-segmentation",
        "beta≈0 & high DSC: (near) perfect segmentation": "beta≈0 & high DSC:\n(near) perfect segmentation",
        "beta≈0 & low DSC: size matches but poor overlap (mis-segmentation)": "beta≈0 & low DSC:\nsize matches but\npoor overlap\n(mis-segmentation)",
        "beta>0: over-segmentation": "beta>0:\nover-segmentation",
        "beta<0: under-segmentation": "beta<0:\nunder-segmentation"
    }
    
    ax = plt.gca()
    
    new_labels = []
    for cat in x_order:
        short = short_labels.get(cat, cat)
        n = counts.get(cat, 0)
        new_labels.append(f"{short}\n(n={n})")
    
    ax.set_xticklabels(new_labels)
    
    # ax.set_xticklabels(
    #     [short_labels.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()]
    # )
    
    
    plt.tight_layout()
    plt.savefig(f"{organ_name}_volume_E_with_outliers.png", dpi=300, bbox_inches="tight")
    plt.show()