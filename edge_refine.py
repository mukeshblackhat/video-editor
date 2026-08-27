"""
Edge-aware mask refinement and color decontamination for video compositing.

Replaces the simple Gaussian blur feathering with:
  1. Morphological cleanup to pull mask edges inward (removing fringe artifacts)
  2. Edge-aware feathering using guided filter (or bilateral fallback)
  3. Distance-based alpha gradient in the mask boundary band
  4. Color decontamination to suppress original background color spill in
     semi-transparent edge pixels (hair fringes, fine details)

All functions operate on single frames and are independently testable.
Dependencies: cv2, numpy. Optionally uses cv2.ximgproc for guided filtering.
"""

import cv2
import numpy as np

# Probe for cv2.ximgproc availability once at import time.
_HAS_GUIDED_FILTER: bool = hasattr(cv2, "ximgproc") and hasattr(
    getattr(cv2, "ximgproc", None), "guidedFilter"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def refine_mask_edges(
    mask: np.ndarray,
    frame: np.ndarray,
    blur_radius: int = 5,
    edge_width: int = 15,
) -> np.ndarray:
    """Create a soft, edge-aware alpha matte from a binary segmentation mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask, dtype uint8, shape (H, W). Values are 0 (background) or
        255 (foreground).
    frame : np.ndarray
        Original video frame, dtype uint8, shape (H, W, 3), BGR color order.
    blur_radius : int
        Controls the overall softness of the feathered edge.  Passed through
        to the guided / bilateral / Gaussian filter.  Must be >= 0.
    edge_width : int
        Half-width (in pixels) of the transition band around the mask
        boundary where alpha falls off from 1 to 0.  Larger values give a
        wider, softer gradient.

    Returns
    -------
    np.ndarray
        Float32 alpha mask, shape (H, W), values in [0.0, 1.0].
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must be (H,W,3), got shape {frame.shape}")
    if mask.shape[:2] != frame.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape} does not match frame shape {frame.shape[:2]}"
        )

    # --- Step 0: Binarise (safety) -------------------------------------------
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # --- Step 1: Morphological cleanup ----------------------------------------
    # Slight erosion (2-3 px) pulls the mask inward, trimming the thin fringe
    # of original-background pixels that SAM often includes at object edges.
    erode_size = 3  # 3x3 kernel => ~1.5 px effective erosion on each side
    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_size, erode_size)
    )
    cleaned = cv2.erode(binary, erode_kernel, iterations=1)

    # Small morphological open to remove isolated specks that survived erosion.
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)

    # --- Step 2: Edge-aware feathering ----------------------------------------
    # Convert cleaned mask to float in [0, 1].
    alpha = cleaned.astype(np.float32) / 255.0

    if blur_radius > 0:
        alpha = _edge_aware_feather(alpha, frame, blur_radius)

    # --- Step 3: Distance-based gradient in the boundary band -----------------
    if edge_width > 0:
        alpha = _apply_boundary_gradient(cleaned, alpha, edge_width)

    # Final clamp to [0, 1] for safety.
    np.clip(alpha, 0.0, 1.0, out=alpha)
    return alpha


def decontaminate_edges(
    frame: np.ndarray,
    mask_alpha: np.ndarray,
    bg_color_estimate: tuple | None = None,
    strength: float = 0.7,
) -> np.ndarray:
    """Suppress original-background color spill in semi-transparent edge pixels.

    When a foreground subject is lifted from its original background, pixels
    along the boundary (especially hair fringes and fine detail) contain a
    blend of foreground and original-background color.  This function shifts
    those edge pixels away from the estimated original-background color toward
    the nearby foreground interior color, reducing visible color fringing
    after compositing onto a new background.

    Parameters
    ----------
    frame : np.ndarray
        Original video frame, uint8, (H, W, 3), BGR.
    mask_alpha : np.ndarray
        Soft alpha matte, float32, (H, W), values in [0, 1].
    bg_color_estimate : tuple or None
        BGR color tuple (B, G, R) of the estimated original background.  If
        *None*, it is estimated automatically from pure-background regions
        (alpha < 0.05).
    strength : float
        Blending strength in [0, 1].  1.0 = full decontamination, 0.0 = no
        change.

    Returns
    -------
    np.ndarray
        Decontaminated frame, uint8, (H, W, 3), BGR.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must be (H,W,3), got shape {frame.shape}")
    if mask_alpha.ndim != 2:
        raise ValueError(
            f"mask_alpha must be 2-D, got shape {mask_alpha.shape}"
        )
    if mask_alpha.shape[:2] != frame.shape[:2]:
        raise ValueError(
            f"mask_alpha shape {mask_alpha.shape} does not match "
            f"frame shape {frame.shape[:2]}"
        )

    strength = float(np.clip(strength, 0.0, 1.0))
    if strength == 0.0:
        return frame.copy()

    # --- Step 1: Identify edge pixels (alpha in (0.1, 0.9)) ------------------
    edge_mask = (mask_alpha > 0.1) & (mask_alpha < 0.9)

    # Bail out early if there are no edge pixels to process.
    if not np.any(edge_mask):
        return frame.copy()

    # --- Step 2: Estimate background color if not provided --------------------
    if bg_color_estimate is None:
        bg_color_estimate = estimate_original_bg_color(frame, mask_alpha)
        # If estimation failed (no background pixels), return untouched.
        if bg_color_estimate is None:
            return frame.copy()

    # --- Step 3: Colour decontamination in LAB space --------------------------
    # Convert frame and bg colour to LAB for perceptually uniform manipulation.
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab).astype(np.float32)

    bg_bgr_arr = np.uint8([[list(bg_color_estimate)]])  # (1,1,3)
    bg_lab = cv2.cvtColor(bg_bgr_arr, cv2.COLOR_BGR2Lab).astype(np.float32)
    bg_lab_vec = bg_lab[0, 0]  # shape (3,)

    # Estimate the *foreground interior* colour as the median of pixels with
    # alpha > 0.9, giving us a reference to shift edge pixels toward.
    fg_mask = mask_alpha > 0.9
    if np.any(fg_mask):
        fg_pixels_lab = frame_lab[fg_mask]  # (N, 3)
        fg_lab_vec = np.median(fg_pixels_lab, axis=0)  # (3,)
    else:
        # No solid foreground -- cannot decontaminate meaningfully.
        return frame.copy()

    # For each edge pixel, compute how much its colour leans toward the
    # original background, then push it toward the foreground interior.
    result_lab = frame_lab.copy()

    edge_pixels = frame_lab[edge_mask]  # (M, 3)
    edge_alphas = mask_alpha[edge_mask]  # (M,)

    # Vector from background to foreground in LAB.
    bg_to_fg = fg_lab_vec - bg_lab_vec  # (3,)
    bg_to_fg_norm = np.linalg.norm(bg_to_fg)

    if bg_to_fg_norm < 1e-6:
        # Background and foreground are nearly the same colour -- nothing to
        # decontaminate.
        return frame.copy()

    # For each edge pixel, project its deviation from the bg onto the
    # bg->fg axis.  Pixels that are close to the bg colour get pushed
    # more toward the fg colour.
    pixel_to_bg = edge_pixels - bg_lab_vec  # (M, 3)

    # Scalar projection onto bg->fg direction.
    direction = bg_to_fg / bg_to_fg_norm  # unit vector (3,)
    projection = np.dot(pixel_to_bg, direction)  # (M,)

    # Normalise projection to [0, 1] range (0 = at bg, 1 = at fg).
    proj_normalised = np.clip(projection / bg_to_fg_norm, 0.0, 1.0)  # (M,)

    # Contamination factor: pixels with low projection (close to bg colour)
    # and low alpha have the most contamination.
    contamination = (1.0 - proj_normalised) * (1.0 - edge_alphas)  # (M,)

    # Shift: move the pixel toward what the foreground colour *would* be at
    # that alpha level.  We blend between the current pixel colour and a
    # "decontaminated" version that replaces the bg component with fg.
    decontaminated = edge_pixels + np.outer(
        contamination * strength, bg_to_fg
    )

    # Clamp LAB values to valid ranges (L: 0-255 in OpenCV's uint8 LAB,
    # but we are in float32 here; a/b are 0-255 centered at 128).
    decontaminated[:, 0] = np.clip(decontaminated[:, 0], 0, 255)
    decontaminated[:, 1] = np.clip(decontaminated[:, 1], 0, 255)
    decontaminated[:, 2] = np.clip(decontaminated[:, 2], 0, 255)

    result_lab[edge_mask] = decontaminated

    # Convert back to BGR uint8 (round, don't truncate).
    result_bgr = cv2.cvtColor(
        np.clip(np.round(result_lab), 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR
    )
    return result_bgr


def estimate_original_bg_color(
    frame: np.ndarray,
    mask: np.ndarray,
) -> tuple | None:
    """Estimate the dominant original-background colour from pure-bg regions.

    Parameters
    ----------
    frame : np.ndarray
        Original video frame, uint8, (H, W, 3), BGR.
    mask : np.ndarray
        Either a binary mask (uint8, values 0/255) or a soft alpha mask
        (float32, values in [0, 1]).

    Returns
    -------
    tuple or None
        (B, G, R) colour tuple (uint8 values), or *None* if there are
        insufficient background pixels to estimate.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must be (H,W,3), got shape {frame.shape}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")

    # Determine background region.
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        bg_region = mask < 0.05
    else:
        # uint8 binary mask: background is 0.
        bg_region = mask < 13  # ~5% of 255

    bg_pixels = frame[bg_region]  # (N, 3)

    if bg_pixels.shape[0] < 10:
        # Not enough background pixels to form a reliable estimate.
        return None

    # Convert to LAB for robust median computation, then convert back.
    # Working in LAB avoids issues with median on hue-wrapping in HSV.
    bg_pixels_reshaped = bg_pixels.reshape(1, -1, 3).astype(np.uint8)
    bg_lab = cv2.cvtColor(bg_pixels_reshaped, cv2.COLOR_BGR2Lab)
    median_lab = np.median(bg_lab[0].astype(np.float32), axis=0)  # (3,)

    # Convert the single median LAB pixel back to BGR.
    median_lab_px = np.clip(np.round(median_lab), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    median_bgr = cv2.cvtColor(median_lab_px, cv2.COLOR_Lab2BGR)
    b, g, r = int(median_bgr[0, 0, 0]), int(median_bgr[0, 0, 1]), int(median_bgr[0, 0, 2])
    return (b, g, r)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _edge_aware_feather(
    alpha: np.ndarray,
    frame: np.ndarray,
    blur_radius: int,
) -> np.ndarray:
    """Apply edge-aware smoothing to the alpha matte.

    Tries cv2.ximgproc.guidedFilter first (best quality -- respects image
    edges like individual hair strands).  Falls back to bilateral filter,
    and finally to Gaussian blur if neither is available / applicable.

    Parameters
    ----------
    alpha : np.ndarray
        Float32 alpha in [0, 1], shape (H, W).
    frame : np.ndarray
        Original frame uint8, (H, W, 3).
    blur_radius : int
        Controls filter kernel size.

    Returns
    -------
    np.ndarray
        Smoothed alpha, float32, (H, W).
    """
    # Use the luminance channel of the frame as the guide image.
    guide_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if _HAS_GUIDED_FILTER:
        # Guided filter parameters:
        #   radius  -- neighbourhood size, derived from blur_radius
        #   eps     -- regularisation; smaller = more edge-sensitive
        radius = max(1, blur_radius * 2)
        eps = 0.01  # low eps => strong edge preservation
        alpha = cv2.ximgproc.guidedFilter(
            guide=guide_gray,
            src=alpha,
            radius=radius,
            eps=eps,
        )
    else:
        # Bilateral filter fallback: preserves edges while smoothing.
        # cv2.bilateralFilter requires uint8 input, so we round-trip.
        alpha_u8 = np.clip(alpha * 255, 0, 255).astype(np.uint8)
        d = max(1, blur_radius * 2 + 1)
        sigma_colour = 75
        sigma_space = max(1, blur_radius * 2)
        alpha_u8 = cv2.bilateralFilter(
            alpha_u8, d, sigma_colour, sigma_space
        )
        alpha = alpha_u8.astype(np.float32) / 255.0

    return alpha


def _apply_boundary_gradient(
    binary_mask: np.ndarray,
    alpha: np.ndarray,
    edge_width: int,
) -> np.ndarray:
    """Create a smooth distance-based gradient in the mask boundary band.

    Within the band between the eroded interior and the dilated exterior,
    alpha transitions smoothly from 1 (inside) to 0 (outside) based on the
    distance transform.

    Parameters
    ----------
    binary_mask : np.ndarray
        Cleaned binary mask, uint8, (H, W), values 0 or 255.
    alpha : np.ndarray
        Current soft alpha, float32, (H, W), values in [0, 1].
    edge_width : int
        Half-width of the transition band in pixels.

    Returns
    -------
    np.ndarray
        Alpha with smooth boundary gradient applied, float32, (H, W).
    """
    ew = max(1, edge_width)

    # Build the edge band: region between erosion and dilation of the mask.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ew * 2 + 1, ew * 2 + 1)
    )
    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    dilated = cv2.dilate(binary_mask, kernel, iterations=1)

    # Edge band = dilated AND NOT eroded.
    band = (dilated > 127) & (eroded < 128)

    # Compute distance from the foreground boundary *into* the foreground
    # (distance from bg pixels).  Pixels deep inside the foreground have
    # high distance.
    # We use the binary mask (cleaned) for the distance transform.
    dist_inside = cv2.distanceTransform(
        binary_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    # Distance from the background boundary into the background.
    dist_outside = cv2.distanceTransform(
        255 - binary_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )

    # In the band, alpha = dist_inside / (dist_inside + dist_outside).
    # This gives a smooth 0->1 falloff based on how close a pixel is to the
    # foreground interior vs. the background exterior.
    total_dist = dist_inside + dist_outside
    # Avoid division by zero (pixels exactly on the boundary).
    safe_total = np.where(total_dist > 0, total_dist, 1.0)
    gradient = dist_inside / safe_total

    # Apply a smooth (cosine) ramp for more natural falloff instead of
    # a simple linear ramp.
    gradient = 0.5 * (1.0 - np.cos(np.pi * gradient))

    # Only apply the gradient within the edge band; keep the alpha from the
    # edge-aware feathering step for regions outside the band.
    result = alpha.copy()
    result[band] = gradient[band].astype(np.float32)

    # Ensure solid interior stays at 1.0 and solid exterior stays at 0.0.
    solid_fg = eroded > 127
    solid_bg = dilated < 128
    result[solid_fg] = 1.0
    result[solid_bg] = 0.0

    return result
