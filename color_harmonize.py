"""
Color Harmonization — Make foreground match the background scene.

When compositing a foreground subject onto a new background, the original
lighting, color temperature, and exposure often clash. This module provides
perceptually-accurate corrections in LAB color space so the subject looks
like it naturally belongs in the new scene.

Functions can be used independently or orchestrated via harmonize_colors().

Dependencies: cv2, numpy (no other packages required).
"""

import cv2
import numpy as np


def harmonize_colors(fg_frame: np.ndarray, bg_frame: np.ndarray,
                     mask_alpha: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """Orchestrate all color harmonization steps on the foreground frame.

    Applies histogram matching, white balance correction, exposure matching,
    and ambient color cast in sequence. Each sub-step has its own strength
    parameter scaled relative to the master strength.

    Args:
        fg_frame: Original foreground frame, uint8 (H, W, 3) BGR.
        bg_frame: Background frame resized to match fg, uint8 (H, W, 3) BGR.
        mask_alpha: Soft alpha mask, float32 (H, W) with values in [0, 1].
                    1.0 = fully foreground, 0.0 = fully background.
        strength: Master strength in [0, 1]. 0.0 = no change, 1.0 = full
                  harmonization. Default 0.6.

    Returns:
        Harmonized foreground frame, uint8 (H, W, 3) BGR.

    Raises:
        ValueError: If inputs have incompatible shapes or invalid dtypes.
    """
    _validate_inputs(fg_frame, bg_frame, mask_alpha, strength)

    if strength <= 0.0:
        return fg_frame.copy()

    # Clamp strength to [0, 1]
    strength = float(np.clip(strength, 0.0, 1.0))

    # Step 1: Match LAB histogram (color distribution alignment)
    result = match_histogram_lab(fg_frame, bg_frame, mask_alpha,
                                 strength=strength)

    # Step 2: White balance correction (color temperature)
    result = match_white_balance(result, bg_frame, mask_alpha,
                                 strength=strength * 0.5)

    # Step 3: Exposure matching (luminance alignment)
    result = match_exposure(result, bg_frame, mask_alpha,
                            strength=strength * 0.5)

    # Step 4: Ambient color cast on edges (environmental light bounce)
    result = apply_ambient_color_cast(result, bg_frame, mask_alpha,
                                      strength=strength * 0.3)

    return result


def match_histogram_lab(fg: np.ndarray, bg: np.ndarray, mask: np.ndarray,
                        strength: float = 0.6) -> np.ndarray:
    """Match foreground color distribution to background in LAB space.

    For each LAB channel, computes the mean and standard deviation of the
    foreground region (mask > 0.5) and the background region (mask < 0.5),
    then shifts the foreground statistics toward the background statistics
    by the given strength.

    The transfer formula per channel:
        fg_new = (fg - fg_mean) * (bg_std / fg_std) * s + fg_mean * (1-s) + bg_mean * s

    where s = strength. This simultaneously matches the spread (std) and
    center (mean) of the color distributions.

    Args:
        fg: Foreground frame, uint8 (H, W, 3) BGR.
        bg: Background frame, uint8 (H, W, 3) BGR.
        mask: Soft alpha mask, float32 (H, W) in [0, 1].
        strength: Blend strength in [0, 1]. Default 0.6.

    Returns:
        Color-matched foreground, uint8 (H, W, 3) BGR.
    """
    if strength <= 0.0:
        return fg.copy()

    strength = float(np.clip(strength, 0.0, 1.0))

    # Convert to LAB (float64 for precision)
    fg_lab = cv2.cvtColor(fg, cv2.COLOR_BGR2LAB).astype(np.float64)
    bg_lab = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float64)

    # Build boolean masks for foreground and background regions
    fg_mask = mask > 0.5  # (H, W) bool
    bg_mask = mask < 0.5  # (H, W) bool

    # Need enough pixels in each region to compute meaningful stats
    fg_pixel_count = np.count_nonzero(fg_mask)
    bg_pixel_count = np.count_nonzero(bg_mask)

    if fg_pixel_count < 10 or bg_pixel_count < 10:
        # Not enough pixels in one region — skip to avoid degenerate stats
        return fg.copy()

    result_lab = fg_lab.copy()

    for ch in range(3):
        fg_channel = fg_lab[:, :, ch]
        bg_channel = bg_lab[:, :, ch]

        # Stats for foreground pixels
        fg_vals = fg_channel[fg_mask]
        fg_mean = np.mean(fg_vals)
        fg_std = np.std(fg_vals)

        # Stats for background pixels
        bg_vals = bg_channel[bg_mask]
        bg_mean = np.mean(bg_vals)
        bg_std = np.std(bg_vals)

        # Avoid division by zero — if fg has no variance, skip scaling
        if fg_std < 1e-6:
            fg_std = 1e-6

        # Apply the histogram transfer only to foreground pixels
        # fg_new = (fg - fg_mean) * (bg_std / fg_std) * s + fg_mean * (1-s) + bg_mean * s
        std_ratio = bg_std / fg_std
        transferred = ((fg_channel - fg_mean) * std_ratio * strength
                       + fg_mean * (1.0 - strength)
                       + bg_mean * strength)

        # Only modify foreground region
        result_lab[:, :, ch] = np.where(fg_mask, transferred, fg_channel)

    # Clip to valid LAB ranges:
    #   L: [0, 255] in OpenCV's uint8 LAB (maps to [0, 100] perceptual)
    #   A: [0, 255] in OpenCV's uint8 LAB (128 = neutral)
    #   B: [0, 255] in OpenCV's uint8 LAB (128 = neutral)
    result_lab = np.clip(result_lab, 0.0, 255.0)

    # Convert back to BGR uint8
    result_bgr = cv2.cvtColor(
        np.clip(np.round(result_lab), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Preserve original pixels outside foreground to avoid LAB roundtrip
    # quantization error (up to +/-1 per channel)
    fg_mask_3d = fg_mask[:, :, np.newaxis]
    result_bgr = np.where(fg_mask_3d, result_bgr, fg)

    return result_bgr


def match_white_balance(fg: np.ndarray, bg: np.ndarray, mask: np.ndarray,
                        strength: float = 0.5) -> np.ndarray:
    """Correct foreground white balance to match the background scene.

    Estimates the color temperature of each region by looking at the average
    color of bright pixels (L > 150 in LAB space). Computes per-channel
    correction multipliers to shift the foreground toward the background's
    white balance.

    Args:
        fg: Foreground frame, uint8 (H, W, 3) BGR.
        bg: Background frame, uint8 (H, W, 3) BGR.
        mask: Soft alpha mask, float32 (H, W) in [0, 1].
        strength: Correction strength in [0, 1]. Default 0.5.

    Returns:
        White-balance corrected frame, uint8 (H, W, 3) BGR.
    """
    if strength <= 0.0:
        return fg.copy()

    strength = float(np.clip(strength, 0.0, 1.0))

    # Get L channel to find bright pixels
    fg_lab = cv2.cvtColor(fg, cv2.COLOR_BGR2LAB)
    bg_lab = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB)

    fg_mask_bool = mask > 0.5
    bg_mask_bool = mask < 0.5

    # Bright pixels: L > 150 (in OpenCV's 0-255 LAB L range)
    fg_bright = (fg_lab[:, :, 0] > 150) & fg_mask_bool
    bg_bright = (bg_lab[:, :, 0] > 150) & bg_mask_bool

    fg_bright_count = np.count_nonzero(fg_bright)
    bg_bright_count = np.count_nonzero(bg_bright)

    # Need enough bright pixels for a reliable estimate
    if fg_bright_count < 20 or bg_bright_count < 20:
        # Fall back to using all pixels in each region if not enough bright ones
        fg_bright = fg_mask_bool
        bg_bright = bg_mask_bool
        fg_bright_count = np.count_nonzero(fg_bright)
        bg_bright_count = np.count_nonzero(bg_bright)
        if fg_bright_count < 10 or bg_bright_count < 10:
            return fg.copy()

    # Average BGR color of bright foreground and background pixels
    fg_float = fg.astype(np.float64)
    bg_float = bg.astype(np.float64)

    fg_avg = np.array([
        np.mean(fg_float[:, :, c][fg_bright]) for c in range(3)
    ])
    bg_avg = np.array([
        np.mean(bg_float[:, :, c][bg_bright]) for c in range(3)
    ])

    # Compute per-channel correction multiplier
    # Avoid division by zero
    fg_avg_safe = np.where(fg_avg < 1.0, 1.0, fg_avg)
    correction = bg_avg / fg_avg_safe

    # Limit correction range to avoid extreme shifts (0.7x to 1.4x)
    correction = np.clip(correction, 0.7, 1.4)

    # Blend correction toward identity (1.0) by (1 - strength)
    correction = 1.0 + (correction - 1.0) * strength

    # Apply correction only to foreground pixels
    result = fg_float.copy()
    fg_mask_3d = fg_mask_bool[:, :, np.newaxis]  # (H, W, 1)

    corrected = fg_float * correction[np.newaxis, np.newaxis, :]  # broadcast (1,1,3)
    result = np.where(fg_mask_3d, corrected, result)

    return np.clip(result, 0, 255).astype(np.uint8)


def match_exposure(fg: np.ndarray, bg: np.ndarray, mask: np.ndarray,
                   strength: float = 0.5) -> np.ndarray:
    """Match foreground exposure (luminance) to the background scene.

    Compares the mean L channel (luminance) of the foreground and background
    regions in LAB space, then shifts the foreground L channel to close the
    gap by the given strength fraction.

    Args:
        fg: Foreground frame, uint8 (H, W, 3) BGR.
        bg: Background frame, uint8 (H, W, 3) BGR.
        mask: Soft alpha mask, float32 (H, W) in [0, 1].
        strength: Correction strength in [0, 1]. Default 0.5.

    Returns:
        Exposure-matched frame, uint8 (H, W, 3) BGR.
    """
    if strength <= 0.0:
        return fg.copy()

    strength = float(np.clip(strength, 0.0, 1.0))

    fg_lab = cv2.cvtColor(fg, cv2.COLOR_BGR2LAB).astype(np.float64)
    bg_lab = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float64)

    fg_mask_bool = mask > 0.5
    bg_mask_bool = mask < 0.5

    fg_pixel_count = np.count_nonzero(fg_mask_bool)
    bg_pixel_count = np.count_nonzero(bg_mask_bool)

    if fg_pixel_count < 10 or bg_pixel_count < 10:
        return fg.copy()

    # Mean luminance of each region
    fg_l_mean = np.mean(fg_lab[:, :, 0][fg_mask_bool])
    bg_l_mean = np.mean(bg_lab[:, :, 0][bg_mask_bool])

    # Shift foreground L toward background L by strength fraction
    l_shift = (bg_l_mean - fg_l_mean) * strength

    # Apply shift only to foreground pixels
    result_lab = fg_lab.copy()
    result_lab[:, :, 0] = np.where(
        fg_mask_bool,
        fg_lab[:, :, 0] + l_shift,
        fg_lab[:, :, 0]
    )

    # Clip L channel to valid range [0, 255]
    result_lab[:, :, 0] = np.clip(result_lab[:, :, 0], 0.0, 255.0)

    result_bgr = cv2.cvtColor(
        np.clip(np.round(result_lab), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Preserve original pixels outside foreground to avoid LAB roundtrip
    # quantization error
    fg_mask_3d = fg_mask_bool[:, :, np.newaxis]
    result_bgr = np.where(fg_mask_3d, result_bgr, fg)

    return result_bgr


def apply_ambient_color_cast(frame: np.ndarray, bg: np.ndarray,
                             mask_alpha: np.ndarray,
                             strength: float = 0.3) -> np.ndarray:
    """Apply ambient color cast from background onto foreground edges.

    Simulates environmental light bounce: surfaces near the edge of the
    foreground subject pick up color from the surrounding background scene.
    The effect is strongest at semi-transparent edge pixels (low alpha) and
    fades toward the interior.

    Args:
        frame: Current foreground frame, uint8 (H, W, 3) BGR.
        bg: Background frame, uint8 (H, W, 3) BGR.
        mask_alpha: Soft alpha mask, float32 (H, W) in [0, 1].
        strength: Tint strength in [0, 1]. Default 0.3.

    Returns:
        Tinted frame, uint8 (H, W, 3) BGR.
    """
    if strength <= 0.0:
        return frame.copy()

    strength = float(np.clip(strength, 0.0, 1.0))

    # Sample dominant background color from the background region (mask < 0.3)
    bg_region_mask = mask_alpha < 0.3
    bg_pixel_count = np.count_nonzero(bg_region_mask)

    if bg_pixel_count < 10:
        return frame.copy()

    bg_float = bg.astype(np.float64)
    dominant_color = np.array([
        np.mean(bg_float[:, :, c][bg_region_mask]) for c in range(3)
    ])  # (3,) BGR

    # Identify edge pixels: 0.1 < alpha < 0.9
    edge_mask = (mask_alpha > 0.1) & (mask_alpha < 0.9)

    if np.count_nonzero(edge_mask) == 0:
        return frame.copy()

    # Compute tint influence: stronger at lower alpha (more transparent),
    # weaker toward interior. Map alpha [0.1, 0.9] -> influence [1.0, 0.0]
    # then scale by strength.
    influence = np.zeros_like(mask_alpha)
    influence[edge_mask] = (1.0 - (mask_alpha[edge_mask] - 0.1) / 0.8) * strength

    # Apply tint: blend frame toward dominant_color weighted by influence
    result = frame.astype(np.float64)
    influence_3d = influence[:, :, np.newaxis]  # (H, W, 1)
    dominant_3d = dominant_color[np.newaxis, np.newaxis, :]  # (1, 1, 3)

    # Only apply where edge_mask is True
    edge_mask_3d = edge_mask[:, :, np.newaxis]
    tinted = result * (1.0 - influence_3d) + dominant_3d * influence_3d
    result = np.where(edge_mask_3d, tinted, result)

    return np.clip(result, 0, 255).astype(np.uint8)


def _validate_inputs(fg_frame: np.ndarray, bg_frame: np.ndarray,
                     mask_alpha: np.ndarray, strength: float) -> None:
    """Validate inputs for harmonize_colors.

    Raises:
        ValueError: If shapes are incompatible, dtypes wrong, or strength
                    is not a finite number.
    """
    if fg_frame.ndim != 3 or fg_frame.shape[2] != 3:
        raise ValueError(
            f"fg_frame must be (H, W, 3) BGR, got shape {fg_frame.shape}")

    if bg_frame.ndim != 3 or bg_frame.shape[2] != 3:
        raise ValueError(
            f"bg_frame must be (H, W, 3) BGR, got shape {bg_frame.shape}")

    if mask_alpha.ndim != 2:
        raise ValueError(
            f"mask_alpha must be (H, W), got shape {mask_alpha.shape}")

    h, w = fg_frame.shape[:2]
    if bg_frame.shape[:2] != (h, w):
        raise ValueError(
            f"bg_frame shape {bg_frame.shape[:2]} does not match "
            f"fg_frame shape {(h, w)}")

    if mask_alpha.shape != (h, w):
        raise ValueError(
            f"mask_alpha shape {mask_alpha.shape} does not match "
            f"frame shape {(h, w)}")

    if fg_frame.dtype != np.uint8:
        raise ValueError(
            f"fg_frame must be uint8, got {fg_frame.dtype}")

    if bg_frame.dtype != np.uint8:
        raise ValueError(
            f"bg_frame must be uint8, got {bg_frame.dtype}")

    if mask_alpha.dtype != np.float32:
        raise ValueError(
            f"mask_alpha must be float32, got {mask_alpha.dtype}")

    if not np.isfinite(strength):
        raise ValueError(f"strength must be finite, got {strength}")
