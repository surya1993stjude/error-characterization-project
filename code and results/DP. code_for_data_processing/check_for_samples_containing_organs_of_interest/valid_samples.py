import os
from pathlib import Path
import pandas as pd
import nibabel as nib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

META_CSV = "Z:\\ResearchHome\\ClusterHome\\ssarka62\\Documents\\data\\lsBART-project\\TotalSegmentator-dataset\\meta.csv"

# TODO: set this to your TotalSegmentator output root
# expected per-case dirs under this root, e.g.:
#   OUT_ROOT/<image_id>/stomach.nii.gz
# or
#   OUT_ROOT/<image_id>/segmentations/stomach.nii.gz
OUT_ROOT = "Z:\\ResearchHome\\ClusterHome\\ssarka62\\Documents\\data\\lsBART-project\\TotalSegmentator-dataset\\Totalsegmentator_dataset_v201_processed\\test\\ts_out_ct"

ORGANS = ["stomach", "liver", "pancreas"]


def load_meta(meta_csv: str) -> pd.DataFrame:
    # your uploaded meta.csv is ';' separated
    return pd.read_csv(meta_csv, sep=";")


def find_mask_path(case_dir: Path, organ: str) -> Path | None:
    """
    Looks for organ mask inside {OUT_ROOT}/{image_id}/... with common patterns.
    Falls back to recursive search within the case folder.
    """
    # Common direct locations
    candidates = [
        case_dir / f"{organ}.nii.gz",
        case_dir / "segmentations" / f"{organ}.nii.gz",
        case_dir / "masks" / f"{organ}.nii.gz",
    ]
    for p in candidates:
        if p.exists():
            return p

    # Fallback: search anywhere inside the case_dir
    hits = list(case_dir.rglob(f"{organ}.nii.gz"))
    if hits:
        # If multiple, pick the shortest path (usually the "right" one)
        hits.sort(key=lambda x: len(str(x)))
        return hits[0]

    return None


def mask_has_any_nonzero(mask_path: Path) -> bool:
    img = nib.load(str(mask_path))
    # If masks are integer (common), using get_fdata is fine; for even more speed you can use dataobj.
    data = img.get_fdata(dtype=np.float32)
    return np.any(data != 0)


def case_has_all_three(image_id: str, out_root: Path) -> tuple[str, dict[str, bool]]:
    case_dir = out_root / str(image_id)
    present = {}

    if not case_dir.exists():
        # If this prints a lot, OUT_ROOT is probably wrong (too deep or too shallow)
        print(f"[WARN] Missing case folder: {case_dir}")
        return image_id, {org: False for org in ORGANS}

    for organ in ORGANS:
        p = find_mask_path(case_dir, organ)
        if p is None:
            present[organ] = False
        else:
            try:
                present[organ] = mask_has_any_nonzero(p)
            except Exception as e:
                print(f"[WARN] {image_id}: cannot read {organ} at {p}: {e}")
                present[organ] = False

    return image_id, present


def main():
    df = load_meta(META_CSV)
    test_ids = df.loc[df["split"] == "test", "image_id"].astype(str).tolist()
    out_root = Path(OUT_ROOT)

    max_workers = min(32, (os.cpu_count() or 8) * 2)
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(case_has_all_three, cid, out_root) for cid in test_ids]
        for fut in as_completed(futs):
            cid, present = fut.result()
            results[cid] = present

    all_three_ids = [cid for cid, pres in results.items() if all(pres[o] for o in ORGANS)]

    print(f"Total test cases: {len(test_ids)}")
    print(f"Cases with non-empty stomach+liver+pancreas: {len(all_three_ids)}")
    print("\nCase IDs (sorted):")
    for cid in sorted(all_three_ids):
        print(cid)


if __name__ == "__main__":
    main()
