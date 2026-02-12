import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk
from skimage.measure import marching_cubes, mesh_surface_area, find_contours
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
from matplotlib.patches import Patch
from pathlib import Path

# NEW: nifti support
import nibabel as nib

# Read in the associated metadata:
meta=pd.read_csv("Z:\ResearchHome\ClusterHome\ssarka62\Documents\data\lsBART-project\TotalSegmentator-dataset\meta.csv", sep=";")
meta_sorted = meta.sort_values(by="image_id", ascending=True)


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
    """
    For 3D: surface area via marching cubes.
    For 2D: perimeter (boundary length) via contour extraction.
    """
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


def beta_score(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0), eps: float = 1e-8) -> float:
    """
    beta = (SA(pred) - SA(gt)) / SA(gt)
    In 2D SA() is perimeter; in 3D SA() is surface area.
    """
    sa_pred = surface_measure(pred, spacing=spacing)
    sa_gt = surface_measure(gt, spacing=spacing)

    # If GT has no boundary/surface (empty GT), beta is undefined.
    if sa_gt < eps:
        if sa_pred < eps:
            return 0.0
        return float("inf")

    return (sa_pred - sa_gt) / (sa_gt + eps)


# ---------------------------
# Interpretation (same for 2D & 3D)
# ---------------------------
def interpret_beta_dice(
    beta: float,
    dice: float,
    dice_high: float = 0.8,
    beta_tol: float = 0.02
) -> str:
    """
    Slide logic:
      beta > 0  => over-segmentation
      beta < 0  => under-segmentation
      beta ≈ 0  => either perfect segmentation OR complete mis-segmentation,
                  disambiguate via Dice
    """
    if np.isinf(beta):
        return "GT empty, prediction non-empty (beta=inf): false positive / over-seg"

    if abs(beta) <= beta_tol:
        if dice >= dice_high:
            return "beta≈0 & high DSC: (near) perfect segmentation"
        else:
            return "beta≈0 & low DSC: size matches but poor overlap (mis-segmentation)"

    if beta > 0:
        return "beta>0: over-segmentation"
    else:
        return "beta<0: under-segmentation"


# ---------------------------
# Plot helpers
# ---------------------------
def plot_2d_pair(gt, pred, title, spacing=(1.0, 1.0)):
    d = dice_coefficient(pred, gt)
    b = beta_score(pred, gt, spacing=spacing)
    interp = interpret_beta_dice(b, d)

    plt.figure(figsize=(6, 6))
    plt.imshow(np.zeros_like(gt), cmap="gray", vmin=0, vmax=1)

    # GT contour (solid)
    plt.contour(gt.astype(float), levels=[0.5], linewidths=2)
    # Pred contour (dashed)
    plt.contour(pred.astype(float), levels=[0.5], linewidths=2, linestyles="--")

    plt.title(f"{title}\nDice={d:.3f}, β={b:.3f}\n{interp}  (2D uses perimeter)")
    plt.axis("off")
    plt.tight_layout()



def plot_3d_mesh_pair(gt, pred, title, spacing=(1.0, 1.0, 1.0)):
    d = dice_coefficient(pred, gt)
    b = beta_score(pred, gt, spacing=spacing)
    interp = interpret_beta_dice(b, d)

    # Guard against both empty
    if gt.sum() == 0 and pred.sum() == 0:
        fig = plt.figure(figsize=(8, 6))
        plt.title(f"{title}\nDice={d:.3f}, β={b:.3f}\n{interp}\n(both empty)")
        plt.axis("off")
        plt.tight_layout()
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    allv_list = []
    legend_handles = []

    # ---- Ground Truth mesh (green) ----
    if gt.sum() > 0:
        v_gt, f_gt, _, _ = marching_cubes(gt.astype(np.float32), 0.5, spacing=spacing)
        mesh_gt = Poly3DCollection(
            v_gt[f_gt],
            alpha=0.35,
            facecolor="tab:green",
            edgecolor="none"
        )
        ax.add_collection3d(mesh_gt)
        allv_list.append(v_gt)
        legend_handles.append(Patch(color="tab:green", label="Ground Truth"))

    # ---- Prediction mesh (red) ----
    if pred.sum() > 0:
        v_pr, f_pr, _, _ = marching_cubes(pred.astype(np.float32), 0.5, spacing=spacing)
        mesh_pr = Poly3DCollection(
            v_pr[f_pr],
            alpha=0.35,
            facecolor="tab:red",
            edgecolor="none"
        )
        ax.add_collection3d(mesh_pr)
        allv_list.append(v_pr)
        legend_handles.append(Patch(color="tab:red", label="Prediction"))

    # ---- Axis limits & aspect ----
    allv = np.vstack(allv_list)
    ax.set_xlim(allv[:, 0].min(), allv[:, 0].max())
    ax.set_ylim(allv[:, 1].min(), allv[:, 1].max())
    ax.set_zlim(allv[:, 2].min(), allv[:, 2].max())
    ax.set_box_aspect((
        np.ptp(allv[:, 0]),
        np.ptp(allv[:, 1]),
        np.ptp(allv[:, 2])
    ))

    # ax.set_xlabel("Z (mm)")
    # ax.set_ylabel("Y (mm)")
    # ax.set_zlabel("X (mm)")
    ax.set_xlabel("Z")
    ax.set_ylabel("Y")
    ax.set_zlabel("X")
    ax.set_title(f"{title}\nDice={d:.3f}, β={b:.3f}\n{interp}")

    # ---- Legend ----
    ax.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()




# ---------------------------
# NEW: NIfTI loading helpers
# ---------------------------
def load_nifti_mask(path: str) -> np.ndarray:
    """
    Loads a .nii (or .nii.gz) mask and returns a numpy array.
    Assumes it's already a binary mask or labelmap; we treat >0 as foreground.
    """
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return (data > 0), img  # return binary + image object


def spacing_zyx_from_nifti(img: nib.Nifti1Image):
    """
    nibabel zooms are typically (x,y,z). We convert to (z,y,x) for marching_cubes spacing.
    """
    sx, sy, sz = img.header.get_zooms()[:3]
    return (float(sz), float(sy), float(sx))


# ---------------------------
# Demo: 2D + 3D examples
# ---------------------------
if __name__ == "__main__":
    # ---- 2D example (unchanged) ----
    H, W = 256, 256
    gt2 = np.zeros((H, W), dtype=bool)
    pred2_over = np.zeros((H, W), dtype=bool)
    pred2_under = np.zeros((H, W), dtype=bool)

    rr, cc = disk((128, 128), 55, shape=gt2.shape)
    gt2[rr, cc] = True

    rr, cc = disk((132, 128), 65, shape=pred2_over.shape)  # larger + slight shift
    pred2_over[rr, cc] = True

    rr, cc = disk((124, 128), 45, shape=pred2_under.shape)  # smaller + slight shift
    pred2_under[rr, cc] = True

    plot_2d_pair(gt2, pred2_over, "2D example: Over-segmentation")
    plot_2d_pair(gt2, pred2_under, "2D example: Under-segmentation")

    # ---- 3D example (NOW FROM .nii FILES) ----
    # Replace these with your file paths:
    GT_NII_PATH = Path(r"Z:\\ResearchHome\\ClusterHome\\ssarka62\\Documents\\data\lsBART-project\\TotalSegmentator-dataset\\Totalsegmentator_dataset_v201_processed\\test\\s0291\\segmentations\\pancreas.nii")
    PRED_NII_PATH = r"Z:\\ResearchHome\\ClusterHome\\ssarka62\\Documents\\data\\lsBART-project\\TotalSegmentator-dataset\\Totalsegmentator_dataset_v201_processed\\test\\ts_out_ct\\s0291\\pancreas.nii"
    patient_id = GT_NII_PATH.parents[1].name
    organ_name = GT_NII_PATH.stem
    
    gt3, gt_img = load_nifti_mask(GT_NII_PATH)
    pred3, pred_img = load_nifti_mask(PRED_NII_PATH)

    if gt3.shape != pred3.shape:
        raise ValueError(f"Shape mismatch: GT {gt3.shape} vs Pred {pred3.shape}. Resample first.")

    # Extract spacing from header (use GT spacing as canonical)
    spacing_zyx = spacing_zyx_from_nifti(gt_img)

    # Title includes actual shapes (extracted, not hardcoded)
    plot_3d_mesh_pair(
        gt3,
        pred3,
        f"Patient id: {patient_id} ({organ_name}) \nshape={gt3.shape}",
        spacing=spacing_zyx
    )
    
    plt.savefig(f"beta-and-dsc-plots/beta-and-dsc-{patient_id}-{organ_name}.jpg", dpi=300, bbox_inches="tight")
    plt.show()