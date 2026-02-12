import numpy as np

def dice_coefficient(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    """
    Dice = 2|P∩G| / (|P|+|G|)
    pred, gt: boolean or {0,1} arrays, same shape
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    inter = np.logical_and(pred, gt).sum()
    p = pred.sum()
    g = gt.sum()

    # Define Dice as 1 when both are empty (common convention)
    if p == 0 and g == 0:
        return 1.0

    return (2.0 * inter) / (p + g + eps)


def surface_measure(mask: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    """
    For 3D: surface area via marching cubes.
    For 2D: perimeter via contour extraction.

    mask: 2D or 3D binary array
    spacing: voxel spacing (z,y,x) for 3D or (y,x) for 2D
    """
    mask = mask.astype(bool)

    if mask.ndim == 3:
        try:
            from skimage.measure import marching_cubes, mesh_surface_area
        except ImportError as e:
            raise ImportError("Install scikit-image: pip install scikit-image") from e

        # marching_cubes expects a scalar field; use mask as 0/1 and level=0.5
        if mask.sum() == 0:
            return 0.0

        verts, faces, _, _ = marching_cubes(
            mask.astype(np.float32),
            level=0.5,
            spacing=spacing  # IMPORTANT: physical units
        )
        return float(mesh_surface_area(verts, faces))

    elif mask.ndim == 2:
        try:
            from skimage.measure import find_contours
        except ImportError as e:
            raise ImportError("Install scikit-image: pip install scikit-image") from e

        if mask.sum() == 0:
            return 0.0

        # Extract contour(s) at the 0.5 level; compute polyline length
        contours = find_contours(mask.astype(np.uint8), level=0.5)

        # spacing for 2D: (y,x)
        sy, sx = spacing if len(spacing) == 2 else (1.0, 1.0)

        perimeter = 0.0
        for c in contours:
            # c is Nx2 in (row=y, col=x) coordinates
            dy = np.diff(c[:, 0]) * sy
            dx = np.diff(c[:, 1]) * sx
            perimeter += np.sum(np.sqrt(dx * dx + dy * dy))
        return float(perimeter)

    else:
        raise ValueError("mask must be 2D or 3D.")


def beta_score(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0), eps: float = 1e-8) -> float:
    """
    beta = (SA(pred)-SA(gt))/SA(gt)
    """
    sa_pred = surface_measure(pred, spacing=spacing)
    sa_gt = surface_measure(gt, spacing=spacing)

    # If GT has zero surface area (empty GT), beta is undefined.
    # Decide a convention: return 0 if both empty, else +inf (all prediction is "extra").
    if sa_gt < eps:
        if sa_pred < eps:
            return 0.0
        return float("inf")

    return (sa_pred - sa_gt) / (sa_gt + eps)


def interpret_beta_dice(beta: float, dice: float, dice_high: float = 0.8, beta_tol: float = 0.0) -> str:
    """
    Matches your slide logic:
      beta>0 => over-seg
      beta<0 => under-seg
      beta≈0 needs Dice to disambiguate (perfect vs mis-seg)

    beta_tol: treat |beta|<=beta_tol as "≈0"
    """
    if np.isinf(beta):
        return "GT empty, prediction non-empty (beta=inf): over-seg / false positive organ"

    if abs(beta) <= beta_tol:
        if dice >= dice_high:
            return "beta≈0 & high Dice: (near) perfect segmentation"
        else:
            return "beta≈0 & low Dice: size matches but poor overlap (mis-segmentation / wrong location/shape)"

    if beta > 0:
        return "over-segmentation (prediction surface larger than GT)"
    else:
        return "under-segmentation (prediction surface smaller than GT)"


# ---------- Example usage ----------
if __name__ == "__main__":
    # pred_mask and gt_mask should be numpy arrays of shape (Z,Y,X) for 3D organs (or (Y,X) for 2D)
    # Values can be {0,1} or boolean.
    pred_mask = np.random.rand(64, 128, 128) > 0.98
    gt_mask   = np.random.rand(64, 128, 128) > 0.98

    # Use real voxel spacing from your imaging metadata!
    spacing_zyx = (2.5, 1.0, 1.0)  # example: (slice_thickness, pixel_spacing_y, pixel_spacing_x)

    dsc = dice_coefficient(pred_mask, gt_mask)
    beta = beta_score(pred_mask, gt_mask, spacing=spacing_zyx)

    print(f"Dice = {dsc:.4f}")
    print(f"Beta = {beta:.4f}")
    print("Interpretation:", interpret_beta_dice(beta, dsc, dice_high=0.8, beta_tol=0.02))
