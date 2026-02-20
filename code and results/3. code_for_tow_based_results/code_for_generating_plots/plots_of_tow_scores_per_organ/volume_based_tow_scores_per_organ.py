import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt


patient_ids = image_ids = [
    "s0013", "s0029", "s0038", "s0040", "s0119",
    "s0230", "s0235", "s0236", "s0244", "s0291",
    "s0308", "s0311", "s0423", "s0440", "s0441",
    "s0450", "s0459", "s0468", "s0470", "s0482",
    "s0499", "s0505", "s0543", "s0561", "s0667",
    "s0687", "s0735", "s0753", "s0923", "s0933",
    "s0994", "s1094", "s1119", "s1121", "s1152",
    "s1174", "s1176", "s1212", "s1240", "s1248",
    "s1249", "s1276", "s1322", "s1347", "s1377",
    "s1386", "s1411", "s1412", "s1413", "s1414",
    "s1415", "s1418", "s1420", "s1421", "s1422",
    "s1423", "s1424", "s1425", "s1426", "s1427",
    "s1428"
]


# ---------------------------
# read csv file with scores
# ---------------------------
CSV_IN  = Path("C:\\Users\\ssarka62\\Documents\\GitHub\\error_characterization_project\\code and results\\3. code_for_tow_based_results\\computing_tow_scores\\computing_volume_tow\\output_file(s)\\Volume_and_surface_scores_with_C_scores_with_rounded_columns_and_tow_volume.csv")


# ---------------------------
# Load CSV
# ---------------------------
df = pd.read_csv(CSV_IN)
df = df[df["patient_id"].isin(patient_ids)]


# ==================================================================================

# ================
# Generate plots:
# ================


organs_list = list(set(df["organ"]))
for organ_name in organs_list:
    df_=df[df["organ"]==organ_name]

    df_long = df_.melt(
        id_vars="interpretation_volume",                          # keep this as x-axis category
        value_vars=['tow_volume_lambda_0.00', 'tow_volume_lambda_0.25', 'tow_volume_lambda_0.50', 'tow_volume_lambda_0.75', 'tow_volume_lambda_1.00'],  # scores --> hue
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
    
    # order=["dice_volume", "dice_volume_z", "beta_volume", "beta_volume_z"]



    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    sns.boxplot(
        data=df_plot,
        x="interpretation_volume",        # your categorical column
        y="Score value",        # numeric values
        hue="Score type",  # Score1..Score4 grouped within each Group
        width=0.7,
        showfliers=False,
        order=x_order
    )
    
    plt.title(f"Volume-based $\\tau$ scores for {organ_name}")
    
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
    plt.savefig(f"{organ_name}_volume_tow_without_outliers.png", dpi=300, bbox_inches="tight")
    plt.show()