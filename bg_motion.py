"""
Background Motion Module — Applies realistic movement to the replacement background.

A static background image looks fake behind a moving foreground subject. This module
transforms the background per-frame to simulate camera motion, parallax, and depth,
making the composite look natural.

All functions operate on the background at its ORIGINAL resolution. The caller
(phase3_composite) is responsible for resizing the result to match frame dimensions.

Dependencies: cv2, numpy only.
"""

import cv2
import numpy as np
import math

# Aspect-ratio mismatch above this fraction is worth warning about. Backgrounds
# within this band produce sub-pixel distortion under any fit mode.
ASPECT_WARN_THRESHOLD = 0.02


def aspect_mismatch(bg_w: int, bg_h: int, frame_w: int, frame_h: int) -> float:
    """Relative difference between a background's aspect ratio and the frame's.

    Args:
        bg_w, bg_h: Background dimensions in pixels.
        frame_w, frame_h: Target frame dimensions in pixels.

    Returns:
        Absolute relative difference, e.g. 0.0 for an exact match and ~2.16
        for a 16:9 background against a 9:16 frame. Scale-free: only the
        ratios matter, not the absolute sizes.
    """
    if bg_h <= 0 or frame_h <= 0 or bg_w <= 0 or frame_w <= 0:
        raise ValueError(
            f"dimensions must be positive, got bg {bg_w}x{bg_h}, "
            f"frame {frame_w}x{frame_h}"
        )
    bg_ar = bg_w / bg_h
    frame_ar = frame_w / frame_h
    return abs(bg_ar - frame_ar) / frame_ar


def fit_background_to_frame(bg_image: np.ndarray, frame_w: int, frame_h: int,
                            mode: str = "cover",
                            interpolation: int = cv2.INTER_LANCZOS4) -> np.ndarray:
    """Resize a background to frame dimensions without distorting its geometry.

    A plain cv2.resize() to the frame size stretches the image whenever the
    aspect ratios differ — a 16:9 background behind a 9:16 video is squeezed
    to roughly a third of its correct width. This function scales uniformly
    instead, so circles stay circular and verticals stay vertical.

    Args:
        bg_image: Background image, uint8 (H, W, 3).
        frame_w: Target width in pixels.
        frame_h: Target height in pixels.
        mode: One of:
            'cover'   - scale to fill, centre-crop the overflow. No blank
                        areas, geometry preserved. The sensible default.
            'contain' - scale so the whole image fits, pad the remainder
                        with black. Geometry preserved, may letterbox.
            'stretch' - legacy anisotropic resize. Fills the frame but
                        distorts. Retained for backward compatibility.
        interpolation: OpenCV interpolation flag used for the resize.

    Returns:
        uint8 (frame_h, frame_w, 3) image. The input is never modified.

    Raises:
        ValueError: On an unknown mode, an empty image, or a non-positive
            target dimension.
    """
    if mode not in ("cover", "contain", "stretch"):
        raise ValueError(
            f"unknown mode {mode!r}; expected 'cover', 'contain', or 'stretch'"
        )
    if bg_image is None or bg_image.size == 0:
        raise ValueError("bg_image must be a non-empty array")
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError(
            f"target dimensions must be positive, got {frame_w}x{frame_h}"
        )

    bg_h, bg_w = bg_image.shape[:2]

    if mode == "stretch":
        return cv2.resize(bg_image, (frame_w, frame_h), interpolation=interpolation)

    # Uniform scale factor. 'cover' takes the larger so the frame is filled;
    # 'contain' takes the smaller so the whole image fits.
    scale_w = frame_w / bg_w
    scale_h = frame_h / bg_h
    scale = max(scale_w, scale_h) if mode == "cover" else min(scale_w, scale_h)

    # Round up so rounding never leaves a one-pixel gap along an edge.
    new_w = max(1, int(math.ceil(bg_w * scale)))
    new_h = max(1, int(math.ceil(bg_h * scale)))
    scaled = cv2.resize(bg_image, (new_w, new_h), interpolation=interpolation)

    if mode == "cover":
        # Centre-crop the overflow down to exactly the frame size.
        x0 = max(0, (new_w - frame_w) // 2)
        y0 = max(0, (new_h - frame_h) // 2)
        cropped = scaled[y0:y0 + frame_h, x0:x0 + frame_w]
        # Guard against a short crop from extreme rounding.
        if cropped.shape[0] != frame_h or cropped.shape[1] != frame_w:
            cropped = cv2.resize(cropped, (frame_w, frame_h),
                                 interpolation=interpolation)
        return np.ascontiguousarray(cropped)

    # contain: centre the scaled image on a black canvas.
    canvas = np.zeros((frame_h, frame_w, bg_image.shape[2]), dtype=bg_image.dtype)
    if new_w > frame_w or new_h > frame_h:
        scaled = cv2.resize(scaled, (min(new_w, frame_w), min(new_h, frame_h)),
                            interpolation=interpolation)
        new_h, new_w = scaled.shape[:2]
    x0 = (frame_w - new_w) // 2
    y0 = (frame_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = scaled
    return canvas


def get_default_motion_config() -> dict:
    """Return a default motion configuration with sensible defaults.

    All motion types are enabled with subtle parameters that work well for
    most video lengths (100-1000 frames) and background resolutions.

    Returns:
        dict: Configuration with keys for each motion effect.
    """
    return {
        "ken_burns": {
            "enabled": True,
            "zoom_start": 1.0,
            "zoom_end": 1.10,
            "pan_x": 0.05,
            "pan_y": 0.03,
        },
        "parallax": {
            "enabled": True,
            "amplitude_x": 8,
            "amplitude_y": 4,
            "period_frames": 120,
        },
        "depth_parallax": {
            "enabled": True,
            "strength": 0.3,
        },
        "motion_blur": {
            "enabled": True,
            "strength": 0.5,
        },
    }


def ken_burns_transform(
    image: np.ndarray,
    frame_idx: int,
    total_frames: int,
    zoom_start: float = 1.0,
    zoom_end: float = 1.15,
    pan_x: float = 0.1,
    pan_y: float = 0.05,
) -> np.ndarray:
    """Apply a slow Ken Burns zoom + pan effect over the video duration.

    Linearly interpolates zoom from zoom_start to zoom_end and pans the
    crop window across the image. The cropped region is resized back to the
    original image dimensions so the output size is always identical to input.

    The function guarantees no black borders by clamping the crop window to
    stay within image bounds. To allow room for zoom+pan, the effective zoom
    is always >= 1.0 (we never zoom out beyond the original extent).

    Args:
        image: Background image, uint8 (H, W, 3).
        frame_idx: Current frame index (0-based).
        total_frames: Total number of frames in the video.
        zoom_start: Zoom factor at frame 0 (1.0 = no zoom).
        zoom_end: Zoom factor at the last frame.
        pan_x: Fraction of image width to pan over the full duration.
        pan_y: Fraction of image height to pan over the full duration.

    Returns:
        Transformed image, uint8 (H, W, 3), same size as input.
    """
    h, w = image.shape[:2]

    # Handle single-frame edge case: just use zoom_start, no pan.
    if total_frames <= 1:
        t = 0.0
    else:
        t = frame_idx / (total_frames - 1)

    # Interpolate zoom — clamp to >= 1.0 to prevent zooming out.
    zoom = zoom_start + (zoom_end - zoom_start) * t
    zoom = max(zoom, 1.0)

    # The crop window size is the original size divided by zoom.
    crop_w = w / zoom
    crop_h = h / zoom

    # Pan offset: at t=0 we start at one edge, at t=1 we end pan_x * w further.
    # The available slack is (w - crop_w) in x and (h - crop_h) in y.
    # We center the crop and then offset by the pan amount.
    slack_x = w - crop_w
    slack_y = h - crop_h

    # Center position for the crop at t=0 (centered in the image).
    # Pan moves from center - half_pan to center + half_pan over duration.
    pan_offset_x = pan_x * w * (t - 0.5)
    pan_offset_y = pan_y * h * (t - 0.5)

    # Crop center: image center + pan offset
    cx = w / 2.0 + pan_offset_x
    cy = h / 2.0 + pan_offset_y

    # Compute crop bounds
    x1 = cx - crop_w / 2.0
    y1 = cy - crop_h / 2.0
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # Clamp to image bounds — shift the window if it goes outside.
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        x1 -= (x2 - w)
        x2 = w
    if y2 > h:
        y1 -= (y2 - h)
        y2 = h

    # Final safety clamp (in case crop is larger than image, which shouldn't
    # happen with zoom >= 1.0, but be safe).
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(w), x2)
    y2 = min(float(h), y2)

    # Convert to integer pixel coords for cropping.
    ix1 = int(round(x1))
    iy1 = int(round(y1))
    ix2 = int(round(x2))
    iy2 = int(round(y2))

    # Ensure at least 1x1 crop
    ix2 = max(ix2, ix1 + 1)
    iy2 = max(iy2, iy1 + 1)

    # Clamp to valid ranges after rounding
    ix1 = max(0, min(ix1, w - 1))
    iy1 = max(0, min(iy1, h - 1))
    ix2 = max(ix1 + 1, min(ix2, w))
    iy2 = max(iy1 + 1, min(iy2, h))

    cropped = image[iy1:iy2, ix1:ix2]

    # Resize back to original dimensions
    result = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return result


def parallax_shift(
    image: np.ndarray,
    frame_idx: int,
    amplitude_x: float = 10.0,
    amplitude_y: float = 5.0,
    period_frames: int = 120,
) -> np.ndarray:
    """Apply sinusoidal drift to simulate subtle camera sway.

    Uses slightly different frequencies for X and Y axes to create a
    natural-looking Lissajous-like motion path rather than a simple
    back-and-forth.

    Args:
        image: Background image, uint8 (H, W, 3).
        frame_idx: Current frame index.
        amplitude_x: Max horizontal shift in pixels.
        amplitude_y: Max vertical shift in pixels.
        period_frames: Period of the X oscillation in frames.

    Returns:
        Shifted image, uint8 (H, W, 3), same size as input.
    """
    # Avoid division by zero if period_frames is unreasonable.
    if period_frames <= 0:
        period_frames = 120

    dx = amplitude_x * math.sin(2.0 * math.pi * frame_idx / period_frames)
    # Y uses 0.7x the frequency for a more organic, non-repeating feel.
    dy = amplitude_y * math.sin(2.0 * math.pi * frame_idx / period_frames * 0.7)

    # Build affine translation matrix: [[1, 0, dx], [0, 1, dy]]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = image.shape[:2]
    shifted = cv2.warpAffine(
        image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101
    )
    return shifted


def depth_parallax(
    image: np.ndarray,
    frame_idx: int,
    mask: np.ndarray,
    total_frames: int,
    strength: float = 0.3,
) -> np.ndarray:
    """Shift background opposite to foreground subject movement for depth effect.

    Computes the foreground center of mass from the mask, measures its
    displacement from the frame center, and translates the background in
    the opposite direction. This mimics how a real camera with depth of field
    shows background shifting when the subject moves.

    If the mask is empty (no foreground detected), no shift is applied.

    Args:
        image: Background image, uint8 (H, W, 3).
        frame_idx: Current frame index (unused directly, kept for interface
            consistency and potential future smoothing).
        mask: Foreground mask at frame dimensions. Can be uint8 (0/255) or
            float32 (0.0/1.0). Shape (H_mask, W_mask) or (H_mask, W_mask, 1).
        total_frames: Total frames (unused directly, kept for interface).
        strength: Multiplier for the displacement (0.0 = no shift, 1.0 = full).

    Returns:
        Shifted image, uint8 (H, W, 3), same size as input.
    """
    bg_h, bg_w = image.shape[:2]

    # Normalize mask to a 2D float array
    if mask is None:
        return image.copy()

    m = mask.copy()
    if m.ndim == 3:
        m = m[:, :, 0]
    m = m.astype(np.float32)

    # If mask is in 0-255 range, normalize to 0-1.
    if m.max() > 1.0:
        m = m / 255.0

    # Compute center of mass of the foreground mask.
    total_mass = m.sum()
    if total_mass < 1.0:
        # No foreground detected — no shift.
        return image.copy()

    mask_h, mask_w = m.shape[:2]
    # Row indices (y) and column indices (x)
    y_coords = np.arange(mask_h, dtype=np.float32)
    x_coords = np.arange(mask_w, dtype=np.float32)

    cx_mask = float(np.dot(m.sum(axis=0), x_coords) / total_mass)
    cy_mask = float(np.dot(m.sum(axis=1), y_coords) / total_mass)

    # Displacement from mask center (in mask pixel space)
    disp_x = cx_mask - mask_w / 2.0
    disp_y = cy_mask - mask_h / 2.0

    # Normalize displacement to fraction of mask dimensions
    frac_x = disp_x / mask_w if mask_w > 0 else 0.0
    frac_y = disp_y / mask_h if mask_h > 0 else 0.0

    # Apply opposite displacement to the background, scaled by strength
    # and expressed in background pixel space.
    dx = -frac_x * bg_w * strength
    dy = -frac_y * bg_h * strength

    # Clamp to prevent extreme shifts (safety for unusual masks)
    max_shift = 0.1  # max 10% of background dimension
    dx = np.clip(dx, -bg_w * max_shift, bg_w * max_shift)
    dy = np.clip(dy, -bg_h * max_shift, bg_h * max_shift)

    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        image, M, (bg_w, bg_h), borderMode=cv2.BORDER_REFLECT_101
    )
    return shifted


def apply_motion_blur(
    image: np.ndarray,
    dx: float,
    dy: float,
    strength: float = 0.5,
) -> np.ndarray:
    """Apply directional motion blur along the displacement vector.

    Creates a linear blur kernel oriented in the direction of (dx, dy) and
    applies it to the image. Only applies blur if the displacement magnitude
    exceeds 1 pixel to avoid unnecessary processing.

    Args:
        image: Background image, uint8 (H, W, 3).
        dx: Horizontal displacement in pixels.
        dy: Vertical displacement in pixels.
        strength: Multiplier for blur kernel length (0.0-1.0+).

    Returns:
        Blurred image, uint8 (H, W, 3), same size as input.
    """
    magnitude = math.sqrt(dx * dx + dy * dy)

    # Don't blur if displacement is negligible.
    if magnitude <= 1.0:
        return image.copy()

    # Kernel size proportional to displacement * strength.
    # Ensure odd kernel size and a reasonable minimum/maximum.
    kernel_size = int(round(magnitude * strength))
    kernel_size = max(3, kernel_size)
    kernel_size = min(kernel_size, 31)  # Cap to prevent extreme blur
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Build a directional motion blur kernel.
    # Create a line of 1s rotated to the angle of (dx, dy).
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2

    # Angle of the motion vector
    angle = math.atan2(dy, dx)

    # Draw the line through the center of the kernel
    for i in range(kernel_size):
        offset = i - center
        px = int(round(center + offset * math.cos(angle)))
        py = int(round(center + offset * math.sin(angle)))
        if 0 <= px < kernel_size and 0 <= py < kernel_size:
            kernel[py, px] = 1.0

    # Normalize so the kernel sums to 1
    total = kernel.sum()
    if total > 0:
        kernel /= total
    else:
        # Fallback: identity (should not happen)
        return image.copy()

    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


def apply_background_motion(
    bg_image: np.ndarray,
    frame_idx: int,
    total_frames: int,
    mask: np.ndarray,
    motion_config: dict,
) -> np.ndarray:
    """Apply all enabled motion effects to the background image for one frame.

    This is the main entry point, called once per frame during compositing.
    Effects are applied in a deliberate order:
      1. Ken Burns (zoom + pan) — large-scale framing change
      2. Parallax shift (sinusoidal sway) — camera drift simulation
      3. Depth parallax (mask-driven) — depth-based counter-movement
      4. Motion blur — applied last since it depends on accumulated displacement

    Args:
        bg_image: Original background image, uint8 (H, W, 3) at its native
            resolution. NOT yet resized to frame dimensions.
        frame_idx: Current frame index (0-based).
        total_frames: Total number of frames in the video.
        mask: Foreground segmentation mask at FRAME resolution (not bg
            resolution). Can be uint8 (0/255) or float32 (0.0/1.0).
            Shape (H_frame, W_frame) or (H_frame, W_frame, 1).
        motion_config: Configuration dict. See get_default_motion_config()
            for the expected structure. Missing keys are treated as disabled.

    Returns:
        Transformed background image, uint8 (H, W, 3) at the SAME resolution
        as the input bg_image. The caller resizes this to frame dimensions.
    """
    if bg_image is None or bg_image.size == 0:
        raise ValueError("bg_image must be a non-empty numpy array")

    result = bg_image.copy()

    # Track total displacement for motion blur at the end.
    total_dx = 0.0
    total_dy = 0.0

    # --- Ken Burns ---
    kb_cfg = motion_config.get("ken_burns", {})
    if kb_cfg.get("enabled", False):
        result = ken_burns_transform(
            result,
            frame_idx,
            total_frames,
            zoom_start=kb_cfg.get("zoom_start", 1.0),
            zoom_end=kb_cfg.get("zoom_end", 1.15),
            pan_x=kb_cfg.get("pan_x", 0.1),
            pan_y=kb_cfg.get("pan_y", 0.05),
        )
        # Estimate Ken Burns displacement for motion blur.
        # Approximate: the pan creates displacement between consecutive frames.
        if total_frames > 1:
            per_frame_dx = kb_cfg.get("pan_x", 0.1) * result.shape[1] / (total_frames - 1)
            per_frame_dy = kb_cfg.get("pan_y", 0.05) * result.shape[0] / (total_frames - 1)
            total_dx += per_frame_dx
            total_dy += per_frame_dy

    # --- Parallax Shift ---
    par_cfg = motion_config.get("parallax", {})
    if par_cfg.get("enabled", False):
        amp_x = par_cfg.get("amplitude_x", 10)
        amp_y = par_cfg.get("amplitude_y", 5)
        period = par_cfg.get("period_frames", 120)

        result = parallax_shift(
            result,
            frame_idx,
            amplitude_x=amp_x,
            amplitude_y=amp_y,
            period_frames=period,
        )

        # Estimate per-frame displacement from the sinusoidal motion.
        if period > 0:
            # Derivative of sin at current frame gives instantaneous velocity.
            phase_x = 2.0 * math.pi * frame_idx / period
            phase_y = phase_x * 0.7
            dx_rate = amp_x * 2.0 * math.pi / period * math.cos(phase_x)
            dy_rate = amp_y * 2.0 * math.pi / period * 0.7 * math.cos(phase_y)
            total_dx += dx_rate
            total_dy += dy_rate

    # --- Depth Parallax ---
    dp_cfg = motion_config.get("depth_parallax", {})
    if dp_cfg.get("enabled", False) and mask is not None:
        bg_h, bg_w = result.shape[:2]
        depth_strength = dp_cfg.get("strength", 0.3)

        # We need to estimate the displacement before applying, so we can
        # include it in motion blur. Compute the same center-of-mass logic.
        m = mask.copy()
        if m.ndim == 3:
            m = m[:, :, 0]
        m = m.astype(np.float32)
        if m.max() > 1.0:
            m = m / 255.0
        total_mass = m.sum()
        if total_mass >= 1.0:
            mask_h, mask_w = m.shape[:2]
            y_coords = np.arange(mask_h, dtype=np.float32)
            x_coords = np.arange(mask_w, dtype=np.float32)
            cx_mask = float(np.dot(m.sum(axis=0), x_coords) / total_mass)
            cy_mask = float(np.dot(m.sum(axis=1), y_coords) / total_mass)
            frac_x = (cx_mask - mask_w / 2.0) / mask_w if mask_w > 0 else 0.0
            frac_y = (cy_mask - mask_h / 2.0) / mask_h if mask_h > 0 else 0.0
            dp_dx = -frac_x * bg_w * depth_strength
            dp_dy = -frac_y * bg_h * depth_strength
            max_shift = 0.1
            dp_dx = np.clip(dp_dx, -bg_w * max_shift, bg_w * max_shift)
            dp_dy = np.clip(dp_dy, -bg_h * max_shift, bg_h * max_shift)
            total_dx += dp_dx
            total_dy += dp_dy

        result = depth_parallax(
            result,
            frame_idx,
            mask,
            total_frames,
            strength=depth_strength,
        )

    # --- Motion Blur (applied last, based on accumulated displacement) ---
    mb_cfg = motion_config.get("motion_blur", {})
    if mb_cfg.get("enabled", False):
        result = apply_motion_blur(
            result,
            total_dx,
            total_dy,
            strength=mb_cfg.get("strength", 0.5),
        )

    return result
