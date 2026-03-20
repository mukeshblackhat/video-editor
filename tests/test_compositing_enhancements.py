"""
Comprehensive tests for compositing enhancement modules.

Tests are self-contained: all test data is synthetic (numpy arrays).
No files on disk are required.

Modules under test:
  - edge_refine.py       (mask refinement, color decontamination)
  - color_harmonize.py   (lighting / color harmonization)
  - bg_motion.py         (background movement engine)
  - phase3_composite.py  (integration: composite_frame)
"""

import sys
import os

# Ensure the project root is on the import path so that the modules under test
# can be imported directly (they live at the repo root, not in a package).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import cv2

# --------------------------------------------------------------------------- #
# Modules under test
# --------------------------------------------------------------------------- #
from edge_refine import refine_mask_edges, decontaminate_edges, estimate_original_bg_color
from color_harmonize import (
    harmonize_colors,
    match_histogram_lab,
    match_white_balance,
    match_exposure,
    apply_ambient_color_cast,
    _validate_inputs,
)
from bg_motion import (
    get_default_motion_config,
    ken_burns_transform,
    parallax_shift,
    depth_parallax,
    apply_motion_blur,
    apply_background_motion,
)
from phase3_composite import composite_frame


# --------------------------------------------------------------------------- #
# Shared synthetic data factories
# --------------------------------------------------------------------------- #

def _make_frame(h: int = 200, w: int = 300, color=(120, 80, 200)) -> np.ndarray:
    """Create a solid-color uint8 BGR frame."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


def _make_binary_mask(h: int = 200, w: int = 300, radius: int = 60) -> np.ndarray:
    """Create a uint8 binary mask with a filled circle in the center."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), radius, 255, -1)
    return mask


def _make_soft_alpha(h: int = 200, w: int = 300, radius: int = 60) -> np.ndarray:
    """Create a float32 alpha mask with a soft circle (gradient edges)."""
    # Start from a binary circle, then blur to get soft edges.
    binary = _make_binary_mask(h, w, radius)
    blurred = cv2.GaussianBlur(binary.astype(np.float32), (31, 31), 0)
    return blurred / 255.0


def _make_gradient_frame(h: int = 200, w: int = 300) -> np.ndarray:
    """Create a frame with a horizontal gradient (useful for verifying shifts)."""
    col = np.linspace(0, 255, w, dtype=np.uint8)
    row = np.tile(col, (h, 1))
    frame = np.stack([row, row, row], axis=-1)
    return frame


# =========================================================================== #
# Tests for edge_refine.py
# =========================================================================== #


class TestRefinesMaskEdges:
    """Tests for refine_mask_edges()."""

    def test_refine_mask_produces_soft_alpha(self):
        """Binary mask in -> float32 alpha out with values between 0 and 1 (not just 0/1).

        The bilateral filter (edge-aware feathering step) produces non-binary
        alpha values at the mask boundary. We disable the boundary gradient
        step (edge_width=0) to isolate this effect, since the gradient step
        can overwrite intermediate values.
        """
        h, w = 200, 300
        mask = _make_binary_mask(h, w, radius=60)
        # Use a noisy frame to give the bilateral filter texture to work with.
        np.random.seed(42)
        frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

        # edge_width=0 disables the boundary gradient so the bilateral filter
        # output is preserved.
        alpha = refine_mask_edges(mask, frame, blur_radius=10, edge_width=0)

        assert alpha.dtype == np.float32
        assert alpha.min() >= 0.0
        assert alpha.max() <= 1.0
        # The bilateral filter should produce at least some non-binary values
        # at the mask boundary (pixels that are not exactly 0.0 or 1.0).
        not_binary = (alpha > 0.0) & (alpha < 1.0)
        assert np.count_nonzero(not_binary) > 0, (
            "Expected some non-binary alpha values from edge-aware feathering"
        )

    def test_refine_mask_preserves_solid_interior(self):
        """Interior pixels (well inside the mask) should stay at 1.0."""
        mask = _make_binary_mask(radius=80)
        frame = _make_frame()

        alpha = refine_mask_edges(mask, frame, blur_radius=3, edge_width=10)

        # The very center of a radius-80 circle should remain at 1.0.
        h, w = mask.shape
        center_region = alpha[h // 2 - 10 : h // 2 + 10, w // 2 - 10 : w // 2 + 10]
        assert np.all(center_region == 1.0), (
            f"Center region min={center_region.min()}, expected all 1.0"
        )

    def test_refine_mask_edge_gradient_exists(self):
        """There should be a gradual falloff at the mask boundary, not a hard step.

        The bilateral filter (with edge_width=0) produces non-binary values
        at the boundary. We verify that the transition from background (0)
        to foreground (1) spans multiple pixels — i.e. it is not a single-
        pixel hard step — by counting how many pixels on a horizontal
        scanline through the mask center have values strictly between 0 and 1.
        """
        h, w = 200, 300
        mask = _make_binary_mask(h, w, radius=70)
        # Noisy frame gives the bilateral filter edges to produce a gradient.
        np.random.seed(42)
        frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

        # Disable boundary gradient step to see the bilateral filter output.
        alpha = refine_mask_edges(mask, frame, blur_radius=10, edge_width=0)

        # Sample a horizontal scanline through the center.
        scanline = alpha[h // 2, :]

        # Count non-binary pixels on this scanline (strictly between 0 and 1).
        not_binary = (scanline > 0.0) & (scanline < 1.0)
        not_binary_count = np.count_nonzero(not_binary)
        assert not_binary_count >= 2, (
            f"Expected a gradual edge transition with at least 2 non-binary "
            f"pixels on the scanline, but found {not_binary_count}"
        )

    def test_refine_mask_shape_mismatch_raises(self):
        """Mismatched mask / frame spatial dimensions should raise ValueError."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="does not match"):
            refine_mask_edges(mask, frame)


class TestDecontaminateEdges:
    """Tests for decontaminate_edges()."""

    def test_decontaminate_edges_modifies_edge_pixels(self):
        """Edge-region pixels (0.1 < alpha < 0.9) should be modified."""
        # Build a frame where foreground is red, background is green.
        h, w = 200, 300
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (0, 200, 0)  # green bg
        frame[60:140, 100:200] = (0, 0, 200)  # red fg block

        alpha = _make_soft_alpha(h, w, radius=60)

        result = decontaminate_edges(frame, alpha, strength=0.7)

        # Edge pixels should differ from the original frame.
        edge_mask = (alpha > 0.1) & (alpha < 0.9)
        if np.any(edge_mask):
            orig_edge = frame[edge_mask]
            new_edge = result[edge_mask]
            assert not np.array_equal(orig_edge, new_edge), (
                "Expected edge pixels to change after decontamination"
            )

    def test_decontaminate_strength_zero_returns_copy(self):
        """strength=0 should return an unchanged copy of the frame."""
        frame = _make_frame()
        alpha = _make_soft_alpha()

        result = decontaminate_edges(frame, alpha, strength=0.0)

        np.testing.assert_array_equal(result, frame)
        # Ensure it is a copy, not the same object.
        assert result is not frame


class TestEstimateOriginalBgColor:
    """Tests for estimate_original_bg_color()."""

    def test_estimate_bg_color_returns_tuple(self):
        """Should return a (B, G, R) tuple of ints when background pixels exist."""
        h, w = 200, 300
        # Solid green frame with a small foreground blob.
        frame = np.full((h, w, 3), (0, 180, 0), dtype=np.uint8)
        # Float32 mask: tiny foreground in the center.
        mask = np.zeros((h, w), dtype=np.float32)
        mask[90:110, 140:160] = 1.0

        result = estimate_original_bg_color(frame, mask)

        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 3
        # Each element should be an int in [0, 255].
        for val in result:
            assert isinstance(val, int)
            assert 0 <= val <= 255

    def test_estimate_bg_color_all_foreground_returns_none(self):
        """If mask is all-255 (no background), should return None."""
        h, w = 100, 100
        frame = _make_frame(h, w)
        mask = np.full((h, w), 255, dtype=np.uint8)

        result = estimate_original_bg_color(frame, mask)

        assert result is None


# =========================================================================== #
# Tests for color_harmonize.py
# =========================================================================== #


class TestHarmonizeColors:
    """Tests for harmonize_colors() and its sub-functions."""

    def test_harmonize_strength_zero_returns_copy(self):
        """strength=0 should return an exact copy of the foreground frame."""
        fg = _make_frame(color=(100, 50, 150))
        bg = _make_frame(color=(200, 180, 60))
        alpha = _make_soft_alpha()

        result = harmonize_colors(fg, bg, alpha, strength=0.0)

        np.testing.assert_array_equal(result, fg)
        assert result is not fg

    def test_histogram_matching_shifts_stats(self):
        """After histogram matching, fg mean/std should move toward bg values."""
        h, w = 200, 300
        # Dark foreground, bright background.
        fg = np.full((h, w, 3), 60, dtype=np.uint8)
        bg = np.full((h, w, 3), 200, dtype=np.uint8)
        # Mask: left half = foreground, right half = background.
        alpha = np.zeros((h, w), dtype=np.float32)
        alpha[:, : w // 2] = 1.0

        fg_mask_bool = alpha > 0.5

        # Compute fg mean BEFORE matching.
        fg_lab_before = cv2.cvtColor(fg, cv2.COLOR_BGR2LAB).astype(np.float64)
        fg_L_mean_before = np.mean(fg_lab_before[:, :, 0][fg_mask_bool])

        # Compute bg mean.
        bg_lab = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float64)
        bg_mask_bool = alpha < 0.5
        bg_L_mean = np.mean(bg_lab[:, :, 0][bg_mask_bool])

        result = match_histogram_lab(fg, bg, alpha, strength=0.8)

        # Compute fg mean AFTER matching.
        result_lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float64)
        fg_L_mean_after = np.mean(result_lab[:, :, 0][fg_mask_bool])

        # The fg mean should have moved closer to the bg mean.
        gap_before = abs(bg_L_mean - fg_L_mean_before)
        gap_after = abs(bg_L_mean - fg_L_mean_after)
        assert gap_after < gap_before, (
            f"Expected gap to shrink: before={gap_before:.1f}, after={gap_after:.1f}"
        )

    def test_white_balance_preserves_bg_pixels(self):
        """Pixels outside the foreground mask should remain unchanged."""
        h, w = 200, 300
        fg = _make_frame(h, w, color=(100, 80, 150))
        bg = _make_frame(h, w, color=(200, 190, 170))
        alpha = np.zeros((h, w), dtype=np.float32)
        alpha[:, : w // 2] = 1.0  # left = foreground

        result = match_white_balance(fg, bg, alpha, strength=0.5)

        # Background region (right half) should be identical to original.
        bg_region_orig = fg[:, w // 2 :, :]
        bg_region_result = result[:, w // 2 :, :]
        np.testing.assert_array_equal(bg_region_result, bg_region_orig)

    def test_exposure_matching_closes_luminance_gap(self):
        """L-channel gap between fg and bg should shrink after exposure matching."""
        h, w = 200, 300
        # Very dark fg, very bright bg.
        fg = np.full((h, w, 3), 40, dtype=np.uint8)
        bg = np.full((h, w, 3), 220, dtype=np.uint8)
        alpha = np.zeros((h, w), dtype=np.float32)
        alpha[:, : w // 2] = 1.0

        fg_mask = alpha > 0.5
        bg_mask = alpha < 0.5

        fg_lab = cv2.cvtColor(fg, cv2.COLOR_BGR2LAB).astype(np.float64)
        bg_lab = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float64)
        gap_before = abs(
            np.mean(bg_lab[:, :, 0][bg_mask]) - np.mean(fg_lab[:, :, 0][fg_mask])
        )

        result = match_exposure(fg, bg, alpha, strength=0.5)

        result_lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float64)
        gap_after = abs(
            np.mean(bg_lab[:, :, 0][bg_mask]) - np.mean(result_lab[:, :, 0][fg_mask])
        )

        assert gap_after < gap_before, (
            f"Expected luminance gap to shrink: before={gap_before:.1f}, "
            f"after={gap_after:.1f}"
        )

    def test_ambient_cast_only_affects_edges(self):
        """Only edge pixels (0.1 < alpha < 0.9) should change."""
        h, w = 200, 300
        frame = _make_frame(h, w, color=(120, 100, 80))
        bg = _make_frame(h, w, color=(50, 200, 50))
        alpha = _make_soft_alpha(h, w, radius=60)

        result = apply_ambient_color_cast(frame, bg, alpha, strength=0.5)

        # Interior pixels (alpha >= 0.9) should be unchanged.
        interior = alpha >= 0.9
        if np.any(interior):
            np.testing.assert_array_equal(
                result[interior], frame[interior],
                err_msg="Interior pixels should not be modified by ambient cast"
            )

        # Pure background pixels (alpha <= 0.1) should be unchanged.
        exterior = alpha <= 0.1
        if np.any(exterior):
            np.testing.assert_array_equal(
                result[exterior], frame[exterior],
                err_msg="Exterior pixels should not be modified by ambient cast"
            )

    def test_validate_inputs_catches_wrong_dtype(self):
        """Passing an int mask instead of float32 should raise ValueError."""
        fg = _make_frame()
        bg = _make_frame()
        mask_int = np.zeros((200, 300), dtype=np.int32)  # wrong dtype

        with pytest.raises(ValueError, match="float32"):
            _validate_inputs(fg, bg, mask_int, 0.5)

    def test_validate_inputs_catches_shape_mismatch(self):
        """Different spatial shapes for fg, bg, or mask should raise ValueError."""
        fg = _make_frame(200, 300)
        bg = _make_frame(100, 300)  # different height
        alpha = np.zeros((200, 300), dtype=np.float32)

        with pytest.raises(ValueError, match="does not match"):
            _validate_inputs(fg, bg, alpha, 0.5)


# =========================================================================== #
# Tests for bg_motion.py
# =========================================================================== #


class TestKenBurns:
    """Tests for ken_burns_transform()."""

    def test_ken_burns_no_black_borders(self):
        """Output should have no zero (black) pixels at the edges."""
        # Use a non-black image so any introduced black border is detectable.
        image = np.full((200, 300, 3), 128, dtype=np.uint8)

        result = ken_burns_transform(
            image, frame_idx=5, total_frames=10,
            zoom_start=1.0, zoom_end=1.2, pan_x=0.1, pan_y=0.1,
        )

        # Check the 4 border rows/columns for any all-zero pixels.
        top_row = result[0, :, :]
        bottom_row = result[-1, :, :]
        left_col = result[:, 0, :]
        right_col = result[:, -1, :]

        for name, border in [("top", top_row), ("bottom", bottom_row),
                              ("left", left_col), ("right", right_col)]:
            # At least no row/column should be entirely black.
            assert not np.all(border == 0), f"Black border detected at {name}"

    def test_ken_burns_single_frame(self):
        """total_frames=1 should not crash and should return a valid image."""
        image = _make_frame(100, 150, color=(50, 100, 200))

        result = ken_burns_transform(image, frame_idx=0, total_frames=1)

        assert result.shape == image.shape
        assert result.dtype == np.uint8


class TestParallaxShift:
    """Tests for parallax_shift()."""

    def test_parallax_shift_returns_same_shape(self):
        """Output shape must match input shape."""
        image = _make_frame(200, 300)

        result = parallax_shift(image, frame_idx=10, amplitude_x=10, amplitude_y=5)

        assert result.shape == image.shape

    def test_parallax_shift_actually_shifts(self):
        """With amplitude > 0, the output should differ from the input."""
        image = _make_gradient_frame(200, 300)

        # At frame_idx where sin != 0, the image should change.
        # sin(2*pi*30/120) = sin(pi/2) = 1.0 => max shift.
        result = parallax_shift(
            image, frame_idx=30, amplitude_x=10, amplitude_y=5, period_frames=120,
        )

        assert not np.array_equal(result, image), (
            "Expected shifted output to differ from input"
        )


class TestDepthParallax:
    """Tests for depth_parallax()."""

    def test_depth_parallax_empty_mask_returns_copy(self):
        """All-zero mask (no foreground) should return a copy with no shift."""
        image = _make_gradient_frame(200, 300)
        empty_mask = np.zeros((200, 300), dtype=np.uint8)

        result = depth_parallax(image, frame_idx=5, mask=empty_mask,
                                total_frames=100, strength=0.3)

        np.testing.assert_array_equal(result, image)

    def test_depth_parallax_centered_mask_no_shift(self):
        """A mask centered in the frame should produce near-zero displacement."""
        h, w = 200, 300
        image = _make_gradient_frame(h, w)
        # Centered circle mask.
        mask = _make_binary_mask(h, w, radius=40)

        result = depth_parallax(image, frame_idx=0, mask=mask,
                                total_frames=100, strength=0.5)

        # With a perfectly centered mask, the displacement fraction is ~0,
        # so the result should be very close to the original.
        diff = np.abs(result.astype(np.int16) - image.astype(np.int16))
        # Allow tiny differences from sub-pixel rounding in warpAffine.
        assert diff.mean() < 1.0, (
            f"Centered mask should produce near-zero shift, got mean diff={diff.mean():.2f}"
        )


class TestMotionBlur:
    """Tests for apply_motion_blur()."""

    def test_motion_blur_small_displacement_no_blur(self):
        """Displacement <= 1px should return an identical copy (no blur applied)."""
        image = _make_gradient_frame(100, 150)

        result = apply_motion_blur(image, dx=0.5, dy=0.5, strength=0.5)

        np.testing.assert_array_equal(result, image)


class TestApplyBackgroundMotion:
    """Tests for apply_background_motion() (the orchestrator)."""

    def test_apply_background_motion_all_disabled(self):
        """All effects off should return a copy of the input."""
        image = _make_gradient_frame(200, 300)
        mask = _make_binary_mask(200, 300)
        config = {
            "ken_burns": {"enabled": False},
            "parallax": {"enabled": False},
            "depth_parallax": {"enabled": False},
            "motion_blur": {"enabled": False},
        }

        result = apply_background_motion(image, frame_idx=5, total_frames=50,
                                         mask=mask, motion_config=config)

        np.testing.assert_array_equal(result, image)

    def test_apply_background_motion_all_enabled(self):
        """All effects enabled should produce output without crashing."""
        image = _make_frame(200, 300, color=(100, 150, 200))
        mask = _make_binary_mask(200, 300)
        config = get_default_motion_config()

        result = apply_background_motion(image, frame_idx=10, total_frames=100,
                                         mask=mask, motion_config=config)

        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_default_config_has_all_keys(self):
        """Default config must contain keys for all four motion effects."""
        config = get_default_motion_config()

        for key in ["ken_burns", "parallax", "depth_parallax", "motion_blur"]:
            assert key in config, f"Missing key '{key}' in default motion config"
            assert isinstance(config[key], dict), f"config['{key}'] should be a dict"
            assert "enabled" in config[key], (
                f"config['{key}'] should have an 'enabled' key"
            )


# =========================================================================== #
# Tests for integration (phase3_composite.py)
# =========================================================================== #


class TestCompositeFrame:
    """Tests for composite_frame()."""

    def test_composite_frame_output_shape(self):
        """Output should match input frame dimensions."""
        h, w = 200, 300
        frame = _make_frame(h, w, color=(100, 50, 200))
        mask = _make_binary_mask(h, w)
        bg = _make_frame(h, w, color=(30, 180, 60))
        alpha = _make_soft_alpha(h, w, radius=60)

        result = composite_frame(frame, mask, bg, alpha)

        assert result.shape == (h, w, 3)
        assert result.dtype == np.uint8

    def test_composite_frame_proper_rounding(self):
        """Values should be rounded, not truncated (no systematic dark bias).

        If compositing used int truncation (e.g. astype(uint8) without round),
        the mean result would systematically be lower than the expected mean.
        With proper rounding, the maximum per-pixel error is 0.5.
        """
        h, w = 100, 100
        # Two solid frames at 127 and 128 with alpha=0.5 everywhere.
        # The exact float result is 127.5 for every pixel.
        # - Truncation would give 127 everywhere.
        # - Rounding should give 128 everywhere (or 127 for round-half-even).
        frame = np.full((h, w, 3), 127, dtype=np.uint8)
        bg = np.full((h, w, 3), 128, dtype=np.uint8)
        mask = np.full((h, w), 255, dtype=np.uint8)
        alpha = np.full((h, w), 0.5, dtype=np.float32)

        result = composite_frame(frame, mask, bg, alpha)

        # np.round(127.5) = 128 (Python/numpy rounds half-to-even).
        expected_value = int(np.round(127.5))  # 128
        mean_val = result.mean()
        assert abs(mean_val - expected_value) < 0.5, (
            f"Expected mean ~{expected_value} (proper rounding), got {mean_val:.2f}. "
            f"This suggests truncation instead of rounding."
        )
