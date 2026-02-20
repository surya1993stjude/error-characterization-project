import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk
from skimage.measure import marching_cubes, mesh_surface_area, find_contours
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

    if sa_gt < eps:
        if sa_pred < eps:
            return 0.0
        return float("inf")  # GT empty but pred not empty

    return (sa_pred - sa_gt) / (sa_gt + eps)


# ---------------------------
# Plot helpers
# ---------------------------
def plot_2d_pair(gt, pred, title, spacing=(1.0, 1.0)):
    d = dice_coefficient(pred, gt)
    b = beta_score(pred, gt, spacing=spacing)  # perimeter-based in 2D

    plt.figure(figsize=(6, 6))
    plt.imshow(np.zeros_like(gt), cmap="gray", vmin=0, vmax=1)

    # GT contour (solid)
    plt.contour(gt.astype(float), levels=[0.5], linewidths=2)
    # Pred contour (dashed)
    plt.contour(pred.astype(float), levels=[0.5], linewidths=2, linestyles="--")

    plt.title(f"{title}\nDice={d:.3f}, β={b:.3f}  (2D uses perimeter)")
    plt.axis("off")
    plt.tight_layout()


def plot_3d_mesh_pair(gt, pred, title, spacing=(1.0, 1.0, 1.0)):
    d = dice_coefficient(pred, gt)
    b = beta_score(pred, gt, spacing=spacing)  # surface-area-based in 3D

    # Mesh extraction
    v_gt, f_gt, _, _ = marching_cubes(gt.astype(np.float32), 0.5, spacing=spacing)
    v_pr, f_pr, _, _ = marching_cubes(pred.astype(np.float32), 0.5, spacing=spacing)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # GT mesh
    mesh_gt = Poly3DCollection(v_gt[f_gt], alpha=0.25)
    ax.add_collection3d(mesh_gt)

    # Pred mesh
    mesh_pr = Poly3DCollection(v_pr[f_pr], alpha=0.25)
    ax.add_collection3d(mesh_pr)

    # Bounds
    allv = np.vstack([v_gt, v_pr])
    ax.set_xlim(allv[:, 0].min(), allv[:, 0].max())
    ax.set_ylim(allv[:, 1].min(), allv[:, 1].max())
    ax.set_zlim(allv[:, 2].min(), allv[:, 2].max())
    ax.set_box_aspect((np.ptp(allv[:, 0]), np.ptp(allv[:, 1]), np.ptp(allv[:, 2])))

    ax.set_xlabel("Z (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("X (mm)")
    ax.set_title(f"{title}\nDice={d:.3f}, β={b:.3f}")
    plt.tight_layout()


# ---------------------------
# Demo: 2D + 3D examples
# ---------------------------
if __name__ == "__main__":
    # ---- 2D: circles (GT vs over/under predictions) ----
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

    # ---- 3D: ellipsoids (GT vs over/under predictions) ----
    Z, Y, X = 64, 96, 96
    zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X]

    def make_ellipsoid(center, radii):
        cz, cy, cx = center
        rz, ry, rx = radii
        return (((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0

    gt3 = make_ellipsoid(center=(32, 48, 48), radii=(16, 22, 20))
    pred3_over = make_ellipsoid(center=(33, 48, 50), radii=(18, 25, 23))   # bigger
    pred3_under = make_ellipsoid(center=(31, 48, 46), radii=(13, 18, 16))  # smaller

    spacing_zyx = (2.5, 1.0, 1.0)  # example spacing (z,y,x) in mm
    plot_3d_mesh_pair(gt3, pred3_over, "3D example: Over-segmentation", spacing=spacing_zyx)
    plot_3d_mesh_pair(gt3, pred3_under, "3D example: Under-segmentation", spacing=spacing_zyx)

    plt.show()
