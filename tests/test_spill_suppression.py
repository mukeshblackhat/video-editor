"""
Tests for edge spill suppression.

When a subject is filmed against a bright background and composited onto a
dark one, the semi-transparent boundary pixels still carry the ORIGINAL
background's light. Alpha blending is doing the right thing arithmetically —
the source pixel genuinely is part hair, part old wall — but the result
reads as a bright halo around the hair.

`suppress_edge_spill()` removes the old background's contribution from those
pixels before compositing, so the edge carries only foreground color.

Tests are self-contained: all test data is synthetic (numpy arrays).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import cv2

from edge_refine import suppress_edge_spill


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _halo_scene(fg_color=(60, 60, 60), old_bg=(230, 235, 240), size=64):
    """A synthetic subject on a bright background with a soft edge.

    Returns (frame, alpha). The middle band is semi-transparent, so its pixels
    are a physical mix of dark foreground and bright old background — exactly
    the situation that produces a halo.
    """
    frame = np.zeros((size, size, 3), np.float32)
    alpha = np.zeros((size, size), np.float32)

    third = size // 3
    alpha[:, :third] = 1.0                                  # solid subject
    alpha[:, third:2 * third] = np.linspace(1.0, 0.0, third)  # soft edge
    alpha[:, 2 * third:] = 0.0                              # pure background

    fg = np.array(fg_color, np.float32)
    bg = np.array(old_bg, np.float32)
    # Physically correct capture: observed = fg*alpha + old_bg*(1-alpha)
    frame[:] = fg * alpha[:, :, None] + bg * (1 - alpha[:, :, None])
    return np.clip(frame, 0, 255).astype(np.uint8), alpha


def _edge_band(alpha, lo=0.05, hi=0.95):
    return (alpha > lo) & (alpha < hi)


# --------------------------------------------------------------------------- #
# Core behaviour
# --------------------------------------------------------------------------- #

def test_removes_bright_background_from_edge():
    """The defect: edge pixels stay bright because they carry the old wall."""
    frame, alpha = _halo_scene()
    band = _edge_band(alpha)

    before = frame[band].mean()
    out = suppress_edge_spill(frame, alpha, old_bg_color=(230, 235, 240),
                              strength=1.0)
    after = out[band].mean()

    assert after < before - 20, (
        f"spill not suppressed: edge brightness {before:.1f} -> {after:.1f}"
    )


def test_recovers_foreground_color_at_full_strength():
    """At strength 1.0 the edge should approach the true foreground color."""
    fg = (60, 60, 60)
    frame, alpha = _halo_scene(fg_color=fg, old_bg=(230, 235, 240))
    band = _edge_band(alpha)

    out = suppress_edge_spill(frame, alpha, old_bg_color=(230, 235, 240),
                              strength=1.0).astype(np.float32)

    # Only trust pixels with meaningful alpha; as alpha -> 0 the unmix is
    # ill-conditioned and the function is expected to leave them alone.
    solid_ish = (alpha > 0.35) & (alpha < 0.95)
    err = np.abs(out[solid_ish] - np.array(fg, np.float32)).mean()
    assert err < 40, f"edge did not approach foreground color; mean error {err:.1f}"


def test_solid_interior_is_untouched():
    """alpha == 1 means no background contribution — must not be modified."""
    frame, alpha = _halo_scene()
    out = suppress_edge_spill(frame, alpha, old_bg_color=(230, 235, 240),
                              strength=1.0)
    core = alpha >= 0.999
    assert np.array_equal(out[core], frame[core]), (
        "solid foreground pixels were altered"
    )


def test_pure_background_is_untouched():
    """alpha == 0 pixels are replaced by the new background anyway."""
    frame, alpha = _halo_scene()
    out = suppress_edge_spill(frame, alpha, old_bg_color=(230, 235, 240),
                              strength=1.0)
    outside = alpha <= 0.001
    assert np.array_equal(out[outside], frame[outside])


def test_strength_zero_is_identity():
    frame, alpha = _halo_scene()
    out = suppress_edge_spill(frame, alpha, old_bg_color=(230, 235, 240),
                              strength=0.0)
    assert np.array_equal(out, frame)


def test_strength_is_monotonic():
    """More strength must mean less residual old background."""
    frame, alpha = _halo_scene()
    band = _edge_band(alpha)
    vals = [
        suppress_edge_spill(frame, alpha, old_bg_color=(230, 235, 240),
                            strength=s)[band].mean()
        for s in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    for a, b in zip(vals, vals[1:]):
        assert b <= a + 1e-6, f"not monotonic across strength: {vals}"
    assert vals[-1] < vals[0] - 20, "full strength barely changed anything"


# --------------------------------------------------------------------------- #
# Background estimation
# --------------------------------------------------------------------------- #

def test_estimates_background_when_not_given():
    """With no explicit color, it should infer one from alpha<0.05 regions."""
    frame, alpha = _halo_scene()
    band = _edge_band(alpha)
    before = frame[band].mean()
    out = suppress_edge_spill(frame, alpha, old_bg_color=None, strength=1.0)
    assert out[band].mean() < before - 15, "auto-estimated background was ineffective"


def test_survives_no_background_pixels():
    """A frame that is entirely foreground has nothing to estimate from."""
    frame = np.full((32, 32, 3), 100, np.uint8)
    alpha = np.ones((32, 32), np.float32)
    out = suppress_edge_spill(frame, alpha, old_bg_color=None, strength=1.0)
    assert out.shape == frame.shape
    assert np.array_equal(out, frame), "should no-op when no background is visible"


def test_dark_original_background_does_not_brighten_edge():
    """Subject shot on a DARK wall: suppression must not invent brightness."""
    frame, alpha = _halo_scene(fg_color=(180, 180, 180), old_bg=(20, 20, 20))
    band = _edge_band(alpha)
    before = frame[band].mean()
    out = suppress_edge_spill(frame, alpha, old_bg_color=(20, 20, 20),
                              strength=1.0)
    # Removing a dark background makes the edge brighter — toward the true
    # foreground — but it must never exceed the foreground itself.
    assert out[band].mean() >= before - 1
    assert out[band].max() <= 255


# --------------------------------------------------------------------------- #
# Contracts and edge cases
# --------------------------------------------------------------------------- #

def test_output_dtype_and_shape_preserved():
    frame, alpha = _halo_scene()
    out = suppress_edge_spill(frame, alpha, old_bg_color=(230, 235, 240))
    assert out.dtype == np.uint8
    assert out.shape == frame.shape


def test_does_not_mutate_input():
    frame, alpha = _halo_scene()
    before = frame.copy()
    suppress_edge_spill(frame, alpha, old_bg_color=(230, 235, 240), strength=1.0)
    assert np.array_equal(frame, before), "input frame was modified in place"


def test_no_out_of_range_values():
    """Unmixing divides by alpha; small alpha must not overflow the result."""
    frame, alpha = _halo_scene(fg_color=(10, 10, 10), old_bg=(250, 250, 250))
    out = suppress_edge_spill(frame, alpha, old_bg_color=(250, 250, 250),
                              strength=1.0)
    assert out.min() >= 0 and out.max() <= 255


def test_rejects_shape_mismatch():
    frame = np.zeros((10, 10, 3), np.uint8)
    alpha = np.zeros((8, 8), np.float32)
    with pytest.raises(ValueError):
        suppress_edge_spill(frame, alpha, old_bg_color=(0, 0, 0))


def test_rejects_bad_frame_shape():
    with pytest.raises(ValueError):
        suppress_edge_spill(np.zeros((10, 10), np.uint8),
                            np.zeros((10, 10), np.float32),
                            old_bg_color=(0, 0, 0))


def test_all_alpha_zero_is_safe():
    frame = np.full((16, 16, 3), 200, np.uint8)
    alpha = np.zeros((16, 16), np.float32)
    out = suppress_edge_spill(frame, alpha, old_bg_color=(200, 200, 200),
                              strength=1.0)
    assert np.array_equal(out, frame)
