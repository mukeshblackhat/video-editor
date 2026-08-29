"""
Tests for background aspect-ratio fitting.

The pipeline previously resized any background straight to the frame
dimensions with cv2.resize(), which stretches the image whenever the
background's aspect ratio differs from the video's.  A 16:9 background
behind a 9:16 video was squeezed to 56% of its correct width.

These tests pin the behaviour of fit_background_to_frame():

  cover   - scale to fill, centre-crop the overflow (geometry preserved)
  contain - scale to fit entirely, pad the remainder (geometry preserved)
  stretch - legacy anisotropic resize (geometry NOT preserved)

Tests are self-contained: all test data is synthetic (numpy arrays).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import cv2

from bg_motion import fit_background_to_frame, aspect_mismatch


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _circle_image(w: int, h: int, radius: int = 40) -> np.ndarray:
    """Image with a circle at its centre.

    A circle is the clearest probe for anisotropic distortion: if the
    background is stretched, the circle becomes an ellipse and its
    width/height ratio moves away from 1.0.
    """
    img = np.zeros((h, w, 3), np.uint8)
    cv2.circle(img, (w // 2, h // 2), radius, (255, 255, 255), -1)
    return img


def _circle_extent(img: np.ndarray) -> tuple:
    """Return (width, height) in pixels of the white blob in the image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 127)
    if len(xs) == 0:
        return (0, 0)
    return (xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)


# --------------------------------------------------------------------------- #
# Output shape — every mode must produce exactly the frame dimensions
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mode", ["cover", "contain", "stretch"])
@pytest.mark.parametrize("bg_w,bg_h", [
    (1920, 1080),   # landscape background, portrait frame (worst case)
    (1080, 1920),   # already matching
    (1000, 1000),   # square
    (600, 4000),    # extremely tall
])
def test_output_is_always_frame_size(mode, bg_w, bg_h):
    """Whatever the input aspect, output must match the frame exactly."""
    bg = _circle_image(bg_w, bg_h)
    out = fit_background_to_frame(bg, 1080, 1920, mode=mode)
    assert out.shape[:2] == (1920, 1080), (
        f"{mode} on {bg_w}x{bg_h} produced {out.shape[:2]}, expected (1920, 1080)"
    )
    assert out.dtype == np.uint8


# --------------------------------------------------------------------------- #
# The core guarantee: cover must not distort geometry
# --------------------------------------------------------------------------- #

def test_cover_preserves_circle_aspect():
    """A circle must stay circular under cover, even from a 16:9 source.

    This is the regression the whole fix exists to prevent.
    """
    bg = _circle_image(1920, 1080, radius=200)
    out = fit_background_to_frame(bg, 1080, 1920, mode="cover")
    w, h = _circle_extent(out)
    assert w > 0 and h > 0, "circle vanished entirely"
    ratio = w / h
    assert 0.93 < ratio < 1.07, (
        f"cover distorted the circle: extent {w}x{h}, ratio {ratio:.3f}, "
        f"expected ~1.0"
    )


def test_stretch_does_distort_circle():
    """Legacy stretch is expected to distort — this documents why it is not the default."""
    bg = _circle_image(1920, 1080, radius=200)
    out = fit_background_to_frame(bg, 1080, 1920, mode="stretch")
    w, h = _circle_extent(out)
    ratio = w / h
    # 16:9 -> 9:16 squeezes width to ~31% of the correct proportion.
    assert ratio < 0.6, (
        f"stretch unexpectedly preserved aspect (ratio {ratio:.3f}); "
        f"this test documents the distortion that 'cover' fixes"
    )


def test_cover_fills_frame_completely():
    """Cover must leave no blank bars — every row and column has content."""
    bg = np.full((1080, 1920, 3), 200, np.uint8)
    out = fit_background_to_frame(bg, 1080, 1920, mode="cover")
    assert (out > 0).all(), "cover left empty pixels; it must fill the frame"


def test_contain_pads_rather_than_crops():
    """Contain must fit the whole image, so a mismatched aspect leaves padding."""
    bg = np.full((1080, 1920, 3), 200, np.uint8)
    out = fit_background_to_frame(bg, 1080, 1920, mode="contain")
    # A 16:9 image fitted inside a 9:16 frame leaves large empty bands.
    assert (out == 0).any(), "contain should pad a mismatched aspect ratio"


# --------------------------------------------------------------------------- #
# Content preservation
# --------------------------------------------------------------------------- #

def test_cover_keeps_frame_centre():
    """Cover centre-crops, so the source centre must survive in the output centre."""
    bg = np.zeros((1080, 1920, 3), np.uint8)
    cv2.circle(bg, (960, 540), 150, (0, 0, 255), -1)   # red dot dead centre
    out = fit_background_to_frame(bg, 1080, 1920, mode="cover")
    ch, cw = out.shape[0] // 2, out.shape[1] // 2
    assert out[ch, cw, 2] > 200, "cover lost the centre of the source image"


def test_matching_aspect_is_a_plain_resize():
    """When aspects already agree, cover must not crop anything away."""
    bg = _circle_image(540, 960, radius=100)
    out = fit_background_to_frame(bg, 1080, 1920, mode="cover")
    w, h = _circle_extent(out)
    ratio = w / h
    assert 0.95 < ratio < 1.05
    # Circle occupied 100/540 of width; after a pure 2x scale it should still
    # occupy the same *fraction* of the frame.
    assert abs(w / 1080 - 200 / 540) < 0.03, (
        "matching-aspect input was cropped when it should have been scaled only"
    )


# --------------------------------------------------------------------------- #
# aspect_mismatch() — the warning helper
# --------------------------------------------------------------------------- #

def test_aspect_mismatch_zero_when_identical():
    assert aspect_mismatch(1080, 1920, 1080, 1920) == pytest.approx(0.0, abs=1e-6)


def test_aspect_mismatch_detects_landscape_vs_portrait():
    m = aspect_mismatch(1920, 1080, 1080, 1920)
    assert m > 2.0, f"16:9 vs 9:16 should be a huge mismatch, got {m}"


def test_aspect_mismatch_small_for_near_match():
    """The project's real backgrounds are within ~0.5% and must not warn."""
    m = aspect_mismatch(3072, 5432, 2160, 3840)   # bg2.png vs input.mp4
    assert m < 0.02, f"near-matching aspects reported {m}"


# --------------------------------------------------------------------------- #
# Input validation and edge cases
# --------------------------------------------------------------------------- #

def test_rejects_unknown_mode():
    bg = _circle_image(100, 100)
    with pytest.raises(ValueError, match="mode"):
        fit_background_to_frame(bg, 50, 50, mode="squish")


def test_rejects_empty_image():
    with pytest.raises(ValueError):
        fit_background_to_frame(np.zeros((0, 0, 3), np.uint8), 10, 10)


def test_rejects_non_positive_target():
    bg = _circle_image(100, 100)
    with pytest.raises(ValueError):
        fit_background_to_frame(bg, 0, 100)
    with pytest.raises(ValueError):
        fit_background_to_frame(bg, 100, -5)


def test_single_pixel_source_upscales():
    """Degenerate but must not crash."""
    bg = np.full((1, 1, 3), 128, np.uint8)
    out = fit_background_to_frame(bg, 64, 64, mode="cover")
    assert out.shape[:2] == (64, 64)


def test_does_not_mutate_input():
    bg = _circle_image(400, 300)
    before = bg.copy()
    fit_background_to_frame(bg, 200, 200, mode="cover")
    assert np.array_equal(bg, before), "input background was modified in place"


def test_extreme_aspect_does_not_crash():
    """A 40:1 panorama into a tall frame is still just a crop."""
    bg = np.full((100, 4000, 3), 180, np.uint8)
    out = fit_background_to_frame(bg, 1080, 1920, mode="cover")
    assert out.shape[:2] == (1920, 1080)
    assert (out > 0).all()
