import subprocess
from pathlib import Path

def run_ts(input_nii: str, out_dir: str, task: str | None = None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cmd = ["TotalSegmentator", "-i", input_nii, "-o", str(out)]
    if task:
        cmd += ["--task", task]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

# CT example
run_ts("Z:\ResearchHome\ClusterHome\ssarka62\Documents\data\lsBART-project\TotalSegmentator-dataset\Totalsegmentator_dataset_v201\s0000\ct.nii.gz", "ts_out_ct")

# # MR example
# run_ts("mri.nii.gz", "ts_out_mr", task="total_mr")
