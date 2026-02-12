import os
from pathlib import Path
import pandas as pd
import nibabel as nib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

META_CSV = "Z:\\ResearchHome\\ClusterHome\\ssarka62\\Documents\\data\\lsBART-project\\TotalSegmentator-dataset\\meta.csv"

# Example list of image_ids
image_ids = [
    "s0013", "s0029", "s0038", "s0040", "s0119", "s0230", "s0235", "s0236",
    "s0244", "s0291", "s0308", "s0311", "s0423", "s0440", "s0441", "s0450",
    "s0459", "s0468", "s0470", "s0482", "s0499", "s0505", "s0543", "s0561",
    "s0667", "s0687", "s0735", "s0753", "s0923", "s0933", "s0994", "s1094",
    "s1119", "s1121", "s1152", "s1174", "s1176", "s1212", "s1240", "s1248",
    "s1249", "s1276", "s1322", "s1347", "s1377", "s1386", "s1411", "s1412",
    "s1413", "s1414", "s1415", "s1418", "s1420", "s1421", "s1422", "s1423",
    "s1424", "s1425", "s1426", "s1427", "s1428"
]


# Load metadata
df = pd.read_csv(META_CSV, sep=";")

# Make sure types match (important!)
df["image_id"] = df["image_id"].astype(str)

# Filter and retrieve study_type
valid_samples_metadata=subset = df[df["image_id"].isin(image_ids)]
subset = df[df["image_id"].isin(image_ids)][["image_id", "study_type"]]

print(subset)


id_to_study_type = (
    df[df["image_id"].isin(image_ids)]
    .set_index("image_id")["study_type"]
    .to_dict()
)

print(id_to_study_type)