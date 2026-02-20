import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


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
CSV_IN  = Path("C:\\Users\\ssarka62\\Documents\\GitHub\\error_characterization_project\\code and results\\2. code_for_c_score_based_results\\computing_c_scores\output_file(s)\\Volume_and_surface_scores_with_C_scores.csv")


# ---------------------------
# Load CSV
# ---------------------------
df = pd.read_csv(CSV_IN)
df = df[df["patient_id"].isin(patient_ids)]
no_of_patients=len(np.unique(df["patient_id"]))


# ==================================================================================

# ================
# Generate plots:
# ================


organs_list = list(set(df["organ"]))

df_=df.copy()

df_long = df_.melt(
        id_vars="organ",                          # keep this as x-axis category
        value_vars=['C_surface_lambda_0', 'C_surface_lambda_0.25', 'C_surface_lambda_0.5', 'C_surface_lambda_0.75', 'C_surface_lambda_1.0'],  # scores --> hue
        var_name="Score type",
        value_name="Score value"
    )
    
df_long.head()
    

# x_order = ['C_surface_lambda_0', 'C_surface_lambda_0.25', 'C_surface_lambda_0.5', 'C_surface_lambda_0.75', 'C_surface_lambda_1.0']

# counts = (
# df_long.groupby("organ")
# .size()
# .reindex(x_order)  # ensures correct order
# .fillna(0)
# .astype(int))

# counts=counts//5



sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
    
sns.boxplot(
        data=df_long,
        x="organ",        # your categorical column
        y="Score value",        # numeric values
        hue="Score type",  # Score1..Score4 grouped within each Group
        width=0.7,
        # showfliers=False,
        # order=order
    )
    
plt.title(f"SA-based C score distributions across organs $(n={no_of_patients})$")
    
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
plt.savefig("surface_based_with_outliers.png", dpi=300, bbox_inches="tight")
plt.show()