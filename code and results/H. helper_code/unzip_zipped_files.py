import os
import gzip
import shutil
from pathlib import Path


found = 0
decompressed = 0

root = Path(r"Z:\\ResearchHome\\ClusterHome\\ssarka62\\Documents\\data\\lsBART-project\\TotalSegmentator-dataset\\Totalsegmentator_dataset_v201_processed\\test\\ts_out_ct")

for dirpath, _, filenames in os.walk(root, followlinks=True):
    dirpath = Path(dirpath)

    for name in filenames:
        if not name.lower().endswith(".nii.gz"):
            continue

        gz_path = dirpath / name
        out_path = gz_path.with_suffix("")  # -> .nii

        if out_path.exists():
            continue

        print(f"Decompressing {gz_path}")
        with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)