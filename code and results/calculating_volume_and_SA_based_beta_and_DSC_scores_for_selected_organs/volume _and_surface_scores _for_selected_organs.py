import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes, mesh_surface_area, find_contours
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
from matplotlib.patches import Patch
from pathlib import Path
import nibabel as nib
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import re
import seaborn as sns
import matplotlib.pyplot as plt


# ---------------------------
# Metrics
# ---------------------------
def dice_coefficient(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    """Dice = 2|P∩G| / (|P|+|G|). Returns 1 if both masks are empty."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    p = pred.sum()
    g = gt.sum()
    if p == 0 and g == 0:
        return 1.0
    return (2.0 * inter) / (p + g + eps)


def surface_measure(mask: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    """3D: surface area; 2D: perimeter."""
    mask = mask.astype(bool)

    if mask.ndim == 3:
        if mask.sum() == 0:
            return 0.0
        verts, faces, _, _ = marching_cubes(mask.astype(np.float32), level=0.5, spacing=spacing)
        return float(mesh_surface_area(verts, faces))

    if mask.ndim == 2:
        if mask.sum() == 0:
            return 0.0
        contours = find_contours(mask.astype(np.uint8), level=0.5)
        sy, sx = spacing if len(spacing) == 2 else (1.0, 1.0)

        perimeter = 0.0
        for c in contours:
            dy = np.diff(c[:, 0]) * sy
            dx = np.diff(c[:, 1]) * sx
            perimeter += np.sum(np.sqrt(dx * dx + dy * dy))
        return float(perimeter)

    raise ValueError("mask must be 2D or 3D.")


def beta_surface_score(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0), eps: float = 1e-8) -> float:
    """beta_surface = (SA(pred) - SA(gt)) / SA(gt)"""
    sa_pred = surface_measure(pred, spacing=spacing)
    sa_gt = surface_measure(gt, spacing=spacing)

    if sa_gt < eps:
        if sa_pred < eps:
            return 0.0
        return float("inf")

    return (sa_pred - sa_gt) / (sa_gt + eps)


def volume_measure(mask: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    """3D: physical volume; 2D: physical area."""
    mask = mask.astype(bool)

    if mask.ndim == 3:
        if len(spacing) != 3:
            raise ValueError("For 3D masks, spacing must have length 3.")
        voxel_vol = float(spacing[0] * spacing[1] * spacing[2])
        return float(mask.sum() * voxel_vol)

    if mask.ndim == 2:
        if len(spacing) != 2:
            sy, sx = spacing[-2], spacing[-1]
        else:
            sy, sx = spacing
        pixel_area = float(sx * sy)
        return float(mask.sum() * pixel_area)

    raise ValueError("mask must be 2D or 3D.")


def beta_volume_score(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0), eps: float = 1e-8) -> float:
    """beta_volume = (V(pred) - V(gt)) / V(gt)"""
    v_pred = volume_measure(pred, spacing=spacing if pred.ndim == 3 else spacing[-2:])
    v_gt = volume_measure(gt, spacing=spacing if gt.ndim == 3 else spacing[-2:])

    if v_gt < eps:
        if v_pred < eps:
            return 0.0
        return float("inf")

    return (v_pred - v_gt) / (v_gt + eps)


def surface_dice(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing=(1.0, 1.0, 1.0),
    tolerance_mm: float = 1.0
) -> float:
    """Surface Dice at tolerance (mm), using mesh vertices + KD-tree."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return float("nan")
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    v_pred, _, _, _ = marching_cubes(pred.astype(np.float32), 0.5, spacing=spacing)
    v_gt, _, _, _ = marching_cubes(gt.astype(np.float32), 0.5, spacing=spacing)

    if len(v_pred) == 0 and len(v_gt) == 0:
        return float("nan")
    if len(v_pred) == 0 or len(v_gt) == 0:
        return 0.0

    tree_pred = cKDTree(v_pred)
    tree_gt = cKDTree(v_gt)

    d_pred_to_gt, _ = tree_gt.query(v_pred, k=1)
    d_gt_to_pred, _ = tree_pred.query(v_gt, k=1)

    pred_match = np.sum(d_pred_to_gt <= tolerance_mm)
    gt_match = np.sum(d_gt_to_pred <= tolerance_mm)

    return float((pred_match + gt_match) / (len(v_pred) + len(v_gt)))


# ---------------------------
# Interpretation (unchanged)
# ---------------------------
def interpret_beta_dice(
    beta: float,
    dice: float,
    dice_high: float = 0.8,
    beta_tol: float = 0.02
) -> str:
    if np.isinf(beta):
        return "GT empty, prediction non-empty (beta=inf): false positive / over-seg"

    if abs(beta) <= beta_tol:
        if dice >= dice_high:
            return "beta≈0 & high DSC: (near) perfect segmentation"
        else:
            return "beta≈0 & low DSC: size matches but poor overlap (mis-segmentation)"

    return "beta>0: over-segmentation" if beta > 0 else "beta<0: under-segmentation"


# ---------------------------
# NIfTI helpers (kept as your convention)
# ---------------------------
def load_nifti_mask(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return (data > 0), img


def spacing_zyx_from_nifti(img: nib.Nifti1Image):
    sx, sy, sz = img.header.get_zooms()[:3]
    return (float(sz), float(sy), float(sx))


# ---------------------------
# Z-score helper
# ---------------------------
def add_zscores(df: pd.DataFrame, cols: list[str], suffix: str = "_z") -> pd.DataFrame:
    """
    Adds z-score columns for each numeric column in `cols`.
    Uses population-style z = (x - mean) / std with ddof=0.
    Ignores NaNs; returns NaN where std==0 or value missing.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        x = pd.to_numeric(out[c], errors="coerce")
        mu = x.mean(skipna=True)
        sigma = x.std(skipna=True, ddof=0)
        if sigma == 0 or np.isnan(sigma):
            out[c + suffix] = np.nan
        else:
            out[c + suffix] = (x - mu) / sigma
    return out


# ---------------------------
# Batch runner
# ---------------------------
def run_beta_dice_batch(
    patient_ids: list[str],
    organs: list[str],
    gt_root: Path,
    pred_root: Path,
    surface_dice_tolerances_mm: tuple[float, ...] = (1.0,),
    dice_high: float = 0.8,
    beta_tol: float = 0.02,
) -> pd.DataFrame:
    """
    Computes:
      - dice_volume
      - beta_surface
      - beta_volume
      - surface_dice_{tol}mm (for tol in surface_dice_tolerances_mm)
    plus 3 interpretation strings:
      - interpretation_current  = (beta_surface + dice_volume)
      - interpretation_surface  = (beta_surface + surface_dice at first tolerance)
      - interpretation_volume   = (beta_volume  + dice_volume)
    Then adds z-score columns for:
      - dice_volume
      - surface_dice_{first tol}mm (as the canonical surface dice for z-scoring)
      - beta_volume
      - beta_surface
    """
    results: list[dict] = []

    tol0 = float(surface_dice_tolerances_mm[0]) if len(surface_dice_tolerances_mm) else 1.0
    surf_dice_key0 = f"surface_dice_{str(tol0).replace('.', 'p')}mm"
    patient_count=0
    
    for patient_id in patient_ids:
        patient_count+=1
        for organ in organs:
            print(f"Patient number {patient_count}, organ {organ}")
            
            gt_path = gt_root / patient_id / "segmentations" / f"{organ}.nii"
            pred_path = pred_root / patient_id / f"{organ}.nii"

            row = {
                "patient_id": patient_id,
                "organ": organ,
                "gt_path": str(gt_path),
                "pred_path": str(pred_path),
                "status": "ok",
                "error": None,

                "shape_gt": None,
                "shape_pred": None,
                "spacing_zyx": None,

                "dice_volume": None,
                "beta_surface": None,
                "beta_volume": None,

                "gt_voxels": None,
                "pred_voxels": None,
                "gt_volume_mm3": None,
                "pred_volume_mm3": None,

                "interpretation_current": None,
                "interpretation_surface": None,
                "interpretation_volume": None,
            }

            # Pre-create surface dice columns
            for tol in surface_dice_tolerances_mm:
                key = f"surface_dice_{str(tol).replace('.', 'p')}mm"
                row[key] = None

            try:
                if (not gt_path.exists()) or (not pred_path.exists()):
                    row["status"] = "missing_file"
                    missing = []
                    if not gt_path.exists():
                        missing.append("GT")
                    if not pred_path.exists():
                        missing.append("PRED")
                    row["error"] = "Missing: " + ",".join(missing)
                    results.append(row)
                    continue

                gt, gt_img = load_nifti_mask(gt_path)
                pred, _ = load_nifti_mask(pred_path)

                row["shape_gt"] = tuple(gt.shape)
                row["shape_pred"] = tuple(pred.shape)

                if gt.shape != pred.shape:
                    row["status"] = "shape_mismatch"
                    row["error"] = f"Shape mismatch: GT {gt.shape} vs Pred {pred.shape}"
                    results.append(row)
                    continue

                spacing_zyx = spacing_zyx_from_nifti(gt_img)
                row["spacing_zyx"] = spacing_zyx

                d_vol = float(dice_coefficient(pred, gt))
                b_surf = float(beta_surface_score(pred, gt, spacing=spacing_zyx))
                b_vol = float(beta_volume_score(pred, gt, spacing=spacing_zyx))

                row["dice_volume"] = d_vol
                row["beta_surface"] = b_surf
                row["beta_volume"] = b_vol

                row["gt_voxels"] = int(gt.sum())
                row["pred_voxels"] = int(pred.sum())
                row["gt_volume_mm3"] = float(volume_measure(gt, spacing=spacing_zyx))
                row["pred_volume_mm3"] = float(volume_measure(pred, spacing=spacing_zyx))

                # Surface dice(s)
                for tol in surface_dice_tolerances_mm:
                    key = f"surface_dice_{str(tol).replace('.', 'p')}mm"
                    row[key] = float(surface_dice(pred, gt, spacing=spacing_zyx, tolerance_mm=float(tol)))

                # Interpretations
                row["interpretation_current"] = interpret_beta_dice(
                    b_surf, d_vol, dice_high=dice_high, beta_tol=beta_tol
                )

                d_surf0 = row.get(surf_dice_key0, np.nan)
                if d_surf0 is None or (isinstance(d_surf0, float) and np.isnan(d_surf0)):
                    row["interpretation_surface"] = "surface dice undefined (both surfaces empty)"
                else:
                    row["interpretation_surface"] = interpret_beta_dice(
                        b_surf, float(d_surf0), dice_high=dice_high, beta_tol=beta_tol
                    )

                row["interpretation_volume"] = interpret_beta_dice(
                    b_vol, d_vol, dice_high=dice_high, beta_tol=beta_tol
                )

                results.append(row)

            except Exception as e:
                row["status"] = "error"
                row["error"] = repr(e)
                results.append(row)

    df = pd.DataFrame(results)

    # Add z-scores for: volume dice, (canonical) surface dice, volume beta, surface beta
    df = add_zscores(df, ["dice_volume", surf_dice_key0, "beta_volume", "beta_surface"], suffix="_z")

    return df


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # ---- INPUTS ----
    patient_ids = [
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
    organs = ["liver", "stomach", "pancreas"]

    # ---- ROOTS ----
    GT_ROOT = Path(
        r"Z:\\ResearchHome\\ClusterHome\\ssarka62\\Documents\\data\\lsBART-project\\TotalSegmentator-dataset\\Totalsegmentator_dataset_v201_processed\\test"
    )
    PRED_ROOT = Path(
        r"Z:\\ResearchHome\\ClusterHome\\ssarka62\\Documents\\data\\lsBART-project\\TotalSegmentator-dataset\\Totalsegmentator_dataset_v201_processed\\test\\ts_out_ct"
    )

    # ---- Surface Dice tolerances (first tol is used for z-score + interpretation_surface) ----
    SURFACE_DICE_TOLS = (1.0,)  # e.g. (1.0, 2.0)

    # ---- Run ----
    df_results = run_beta_dice_batch(
        patient_ids=patient_ids,
        organs=organs,
        gt_root=GT_ROOT,
        pred_root=PRED_ROOT,
        surface_dice_tolerances_mm=SURFACE_DICE_TOLS,
        dice_high=0.8,
        beta_tol=0.02
    )

    # ---- Save ----
    CSV_OUT = Path("Volume and surface scores for selected organs - test.csv")
    df_results.to_csv(CSV_OUT, index=False)
    print(df_results.head())
    print(f"Saved results to: {CSV_OUT.resolve()}")



# # ======================================================================================



# # Code to plot beta, DSC and z-scores for a particular organ:

# df_long = df_results.melt(
#     id_vars="interpretation_volume",                          # keep this as x-axis category
#     # value_vars=["dice_volume","dice_volume_z","surface_dice_1p0mm","surface_dice_1p0mm_z", "beta_volume", "beta_volume_z", "beta_surface", "beta_surface_z"],  # scores become hue groups
#     value_vars=["dice_volume", "dice_volume_z", "beta_volume", "beta_volume_z"],  # scores become hue groups
#     var_name="Score type",
#     value_name="Score value"
# )

# df_long.head()



# sns.set(style="whitegrid")
# plt.figure(figsize=(12, 6))

# sns.boxplot(
#     data=df_long,
#     x="interpretation_volume",        # your categorical column
#     y="Score value",        # numeric values
#     hue="Score type",  # Score1..Score4 grouped within each Group
#     width=0.7
# )

# plt.title("Liver scores")
# plt.tight_layout()
# plt.show()




