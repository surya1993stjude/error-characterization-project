from pathlib import Path
import subprocess

def run_ts(input_nii: str, out_dir: str, task: str | None = None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cmd = ["TotalSegmentator", "-i", input_nii, "-o", str(out)]
    if task:
        cmd += ["--task", task]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

folder_path="Z:\ResearchHome\ClusterHome\ssarka62\Documents\data\lsBART-project\TotalSegmentator-dataset\Totalsegmentator_dataset_v201_processed"
base_path = Path(folder_path)

for folder in base_path.iterdir():
    if folder.is_dir():
        print(folder.name)      # folder name only
        print(folder)           # full path as Path object
        # CT example
        run_ts(f"Z:\ResearchHome\ClusterHome\ssarka62\Documents\data\lsBART-project\TotalSegmentator-dataset\Totalsegmentator_dataset_v201\{folder.name}\ct.nii.gz", f"ts_out_ct\{folder.name}")
