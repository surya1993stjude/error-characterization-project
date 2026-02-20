import subprocess
from pathlib import Path
import pandas as pd
import os

data_root = Path("Z:\\ResearchHome\\ClusterHome\\ssarka62\\Documents\\data\\lsBART-project\\TotalSegmentator-dataset\\Totalsegmentator_dataset_v201_processed\\train")
out_root = Path("Z:\\ResearchHome\\ClusterHome\\ssarka62\\Documents\\data\\lsBART-project\\TotalSegmentator-dataset\\Totalsegmentator_dataset_v201_processed\\train\\ts_out_ct")

first_level_folders_train = [
    name for name in os.listdir(out_root)
    if os.path.isdir(os.path.join(out_root, name))
]


meta=pd.read_csv("Z:\\ResearchHome\\ClusterHome\\ssarka62\\Documents\\data\\lsBART-project\\TotalSegmentator-dataset\\meta.csv", delimiter=";")
meta=meta[meta['split']=='train']
all_training_patients=meta['image_id']


# # # Train set:
# patient_ids=list( set(all_training_patients) - set(first_level_folders_train) )

# Test set:
patient_ids=[
  "s0295", "s0311", "s0308", "s0291", "s0235", "s0236", "s0244", "s0230",
  "s0029", "s1406", "s0119", "s0040", "s1407", "s0013", "s0753", "s1408",
  "s0687", "s0735", "s0684", "s0667", "s0673", "s0561", "s0543", "s1355",
  "s0440", "s0482", "s0505", "s1409", "s0499", "s0459", "s0441", "s0423",
  "s1410", "s0470", "s0450", "s1411", "s0468", "s1412", "s1413", "s1414",
  "s0038", "s1415", "s1276", "s1347", "s1323", "s1322", "s1240", "s1248",
  "s1249", "s1386", "s1094", "s1195", "s1416", "s1417", "s1192", "s1184",
  "s1176", "s1212", "s1174", "s1152", "s1119", "s1377", "s1418", "s1121",
  "s1112", "s1096", "s1075", "s1057", "s1419", "s1051", "s1053", "s0994",
  "s0951", "s0923", "s0933", "s0907", "s1357", "s0829", "s0802", "s1420",
  "s1421", "s1422", "s1423", "s1424", "s1425", "s1426", "s1427", "s1428",
  "s1429"
]

organs = [
    "stomach"
]

# folders = [p for p in base_dir.iterdir() if p.is_dir()]

for patient_dir in data_root.iterdir():
    try:
        if not patient_dir.is_dir():
            continue
    
        ct = patient_dir / "ct.nii"
        if not ct.exists():
            print(f"Missing CT for {patient_dir.name}, skipping")
            continue
    
        out_dir = out_root / patient_dir.name
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        if out_dir.exists():
            print(f"Already processed {patient_dir.name}")
            continue
    
        cmd = [
            "TotalSegmentator",
            "-i", str(ct),
            "-o", str(out_dir),
            "--roi_subset", *organs,
            "--fast",
            "--device", "cpu"
        ]
    
        print(f"Running {patient_dir.name}")
        subprocess.run(cmd, check=True)
    except:
        pass
