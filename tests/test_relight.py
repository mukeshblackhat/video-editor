"""
Tests for subject relighting.

A composited subject reads as a cutout when its lighting does not agree with
the new background: wrong overall brightness, no bounce light from the room,
and no directional key. These functions adjust the subject's *interior* -
not just its edges - so it looks lit by the scene it was placed into.

Tests are self-contained: all test data is synthetic (numpy arrays).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import cv2

from relight import (
    estimate_light_direction,
    match_subject_exposure,
    apply_ambient_bounce,
    apply_directional_light,
    add_contact_shadow,
    relight_subject,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _subject(size=128, value=180):
    """A centred rectangular subject on a transparent field."""
    frame = np.full((size, size, 3), value, np.uint8)
    alpha = np.zeros((size, size), np.float32)
    alpha[size // 4:3 * size // 4, size // 4:3 * size // 4] = 1.0
    return frame, alpha


def _lit_bg(size=128, bright_side="left", base=200):
    """Background with a clear light gradient on one side."""
    bg = np.full((size, size, 3), base, np.uint8)
    # Subtract more where the light is NOT: darken the far side.
    ramp = np.linspace(0, 60, size).astype(np.float32)
    if bright_side == "right":
        ramp = ramp[::-1]
    bg = np.clip(bg.astype(np.float32) - ramp[None, :, None], 0, 255).astype(np.uint8)
    return bg


def _mean_on(img, alpha, thresh=0.99):
    m = alpha >= thresh
    return img[m].astype(np.float32).mean() if m.any() else float("nan")


# --------------------------------------------------------------------------- #
# estimate_light_direction
# --------------------------------------------------------------------------- #

def test_detects_light_from_left():
    bg = _lit_bg(bright_side="left")
    dx, dy = estimate_light_direction(bg)
    assert dx < -0.05, f"expected leftward light vector, got dx={dx}"


def test_detects_light_from_right():
    bg = _lit_bg(bright_side="right")
    dx, dy = estimate_light_direction(bg)
    assert dx > 0.05, f"expected rightward light vector, got dx={dx}"


def test_uniform_background_gives_weak_direction():
    bg = np.full((64, 64, 3), 180, np.uint8)
    dx, dy = estimate_light_direction(bg)
    assert abs(dx) < 0.05 and abs(dy) < 0.05, "uniform bg should have no strong direction"


def test_direction_is_normalised():
    bg = _lit_bg(bright_side="left")
    dx, dy = estimate_light_direction(bg)
    assert np.hypot(dx, dy) <= 1.0 + 1e-6


# --------------------------------------------------------------------------- #
# match_subject_exposure
# --------------------------------------------------------------------------- #

def test_darkens_subject_that_is_too_bright():
    frame, alpha = _subject(value=240)
    bg = np.full((128, 128, 3), 80, np.uint8)
    before = _mean_on(frame, alpha)
    out = match_subject_exposure(frame, alpha, bg, strength=1.0)
    assert _mean_on(out, alpha) < before - 20


def test_brightens_subject_that_is_too_dark():
    frame, alpha = _subject(value=60)
    bg = np.full((128, 128, 3), 220, np.uint8)
    before = _mean_on(frame, alpha)
    out = match_subject_exposure(frame, alpha, bg, strength=1.0)
    assert _mean_on(out, alpha) > before + 15


def test_exposure_leaves_background_pixels_alone():
    frame, alpha = _subject(value=240)
    bg = np.full((128, 128, 3), 80, np.uint8)
    out = match_subject_exposure(frame, alpha, bg, strength=1.0)
    outside = alpha < 0.01
    assert np.array_equal(out[outside], frame[outside])


def test_exposure_strength_zero_is_identity():
    frame, alpha = _subject()
    bg = np.full((128, 128, 3), 40, np.uint8)
    assert np.array_equal(match_subject_exposure(frame, alpha, bg, strength=0.0), frame)


def test_exposure_never_fully_flattens_subject():
    """Matching must move the subject toward the background, not erase contrast."""
    frame, alpha = _subject(value=240)
    frame[40:60, 40:60] = 100                      # a dark feature inside
    bg = np.full((128, 128, 3), 80, np.uint8)
    out = match_subject_exposure(frame, alpha, bg, strength=1.0)
    inner = out[40:60, 40:60].astype(float).mean()
    outer = out[70:80, 70:80].astype(float).mean()
    assert outer - inner > 20, "internal contrast was destroyed"


# --------------------------------------------------------------------------- #
# apply_ambient_bounce
# --------------------------------------------------------------------------- #

def test_bounce_tints_subject_toward_background_hue():
    frame, alpha = _subject(value=180)
    bg = np.zeros((128, 128, 3), np.uint8)
    bg[:, :, 0] = 200                              # strong blue room
    out = apply_ambient_bounce(frame, alpha, bg, strength=1.0)
    core = alpha >= 0.99
    assert out[core][:, 0].mean() > frame[core][:, 0].mean() + 3, "no blue bounce applied"


def test_bounce_affects_interior_not_only_edges():
    """The whole point: this must reach the subject's interior."""
    frame, alpha = _subject(value=180)
    bg = np.zeros((128, 128, 3), np.uint8)
    bg[:, :, 0] = 220
    out = apply_ambient_bounce(frame, alpha, bg, strength=1.0)
    cy = cx = 64                                    # dead centre of the subject
    assert out[cy, cx, 0] > frame[cy, cx, 0] + 2, "interior pixels were not tinted"


def test_bounce_strength_zero_is_identity():
    frame, alpha = _subject()
    bg = np.full((128, 128, 3), 100, np.uint8)
    assert np.array_equal(apply_ambient_bounce(frame, alpha, bg, strength=0.0), frame)


def test_bounce_leaves_background_alone():
    frame, alpha = _subject()
    bg = np.zeros((128, 128, 3), np.uint8); bg[:, :, 2] = 200
    out = apply_ambient_bounce(frame, alpha, bg, strength=1.0)
    outside = alpha < 0.01
    assert np.array_equal(out[outside], frame[outside])


# --------------------------------------------------------------------------- #
# apply_directional_light
# --------------------------------------------------------------------------- #

def test_directional_light_brightens_facing_side():
    frame, alpha = _subject(value=150)
    # estimate_light_direction returns a vector pointing toward the brighter
    # side, so light on the left is (-1, 0).
    out = apply_directional_light(frame, alpha, (-1.0, 0.0), strength=1.0)
    core = alpha >= 0.99
    ys, xs = np.where(core)
    cx = int(xs.mean())
    left = out[core & (np.arange(128)[None, :] < cx)].astype(float).mean()
    right = out[core & (np.arange(128)[None, :] >= cx)].astype(float).mean()
    assert left > right + 3, f"light from the left did not brighten the left ({left} vs {right})"


def test_directional_light_reverses_with_direction():
    frame, alpha = _subject(value=150)
    a = apply_directional_light(frame, alpha, (-1.0, 0.0), strength=1.0)
    b = apply_directional_light(frame, alpha, (1.0, 0.0), strength=1.0)
    core = alpha >= 0.99
    xs = np.where(core)[1]
    cx = int(xs.mean())
    left_sel = core & (np.arange(128)[None, :] < cx)
    assert a[left_sel].astype(float).mean() > b[left_sel].astype(float).mean()


def test_directional_strength_zero_is_identity():
    frame, alpha = _subject()
    assert np.array_equal(
        apply_directional_light(frame, alpha, (-1.0, 0.0), strength=0.0), frame)


def test_directional_light_leaves_background_alone():
    frame, alpha = _subject()
    out = apply_directional_light(frame, alpha, (-1.0, 0.0), strength=1.0)
    outside = alpha < 0.01
    assert np.array_equal(out[outside], frame[outside])


# --------------------------------------------------------------------------- #
# add_contact_shadow
# --------------------------------------------------------------------------- #

def test_contact_shadow_darkens_near_subject():
    bg = np.full((128, 128, 3), 220, np.uint8)
    _, alpha = _subject()
    out = add_contact_shadow(bg, alpha, strength=1.0, offset=(6, 6), blur=9)
    # Just outside the subject's lower-right, where the shadow is cast.
    assert out[97, 97].astype(float).mean() < bg[97, 97].astype(float).mean() - 5


def test_contact_shadow_does_not_darken_far_background():
    bg = np.full((128, 128, 3), 220, np.uint8)
    _, alpha = _subject()
    out = add_contact_shadow(bg, alpha, strength=1.0, offset=(6, 6), blur=9)
    assert out[2, 2].astype(float).mean() > bg[2, 2].astype(float).mean() - 2


def test_contact_shadow_strength_zero_is_identity():
    bg = np.full((64, 64, 3), 200, np.uint8)
    _, alpha = _subject(size=64)
    assert np.array_equal(add_contact_shadow(bg, alpha, strength=0.0), bg)


# --------------------------------------------------------------------------- #
# relight_subject orchestration
# --------------------------------------------------------------------------- #

def test_relight_runs_end_to_end():
    frame, alpha = _subject(value=240)
    bg = _lit_bg(base=120)
    out = relight_subject(frame, alpha, bg)
    assert out.shape == frame.shape and out.dtype == np.uint8


def test_relight_moves_subject_toward_background_brightness():
    frame, alpha = _subject(value=240)
    bg = np.full((128, 128, 3), 90, np.uint8)
    before = abs(_mean_on(frame, alpha) - 90)
    out = relight_subject(frame, alpha, bg)
    after = abs(_mean_on(out, alpha) - 90)
    assert after < before, "relighting did not close the brightness gap"


def test_relight_does_not_mutate_input():
    frame, alpha = _subject()
    bg = _lit_bg()
    before = frame.copy()
    relight_subject(frame, alpha, bg)
    assert np.array_equal(frame, before)


def test_relight_all_disabled_is_identity():
    frame, alpha = _subject()
    bg = _lit_bg()
    out = relight_subject(frame, alpha, bg, exposure=0.0, bounce=0.0, directional=0.0)
    assert np.array_equal(out, frame)


def test_relight_output_in_range():
    frame, alpha = _subject(value=250)
    bg = np.zeros((128, 128, 3), np.uint8)
    out = relight_subject(frame, alpha, bg, exposure=1.0, bounce=1.0, directional=1.0)
    assert out.min() >= 0 and out.max() <= 255


def test_relight_rejects_shape_mismatch():
    frame, alpha = _subject()
    with pytest.raises(ValueError):
        relight_subject(frame, alpha, np.zeros((64, 64, 3), np.uint8))


def test_relight_handles_empty_subject():
    frame = np.full((64, 64, 3), 150, np.uint8)
    alpha = np.zeros((64, 64), np.float32)
    out = relight_subject(frame, alpha, np.full((64, 64, 3), 100, np.uint8))
    assert np.array_equal(out, frame), "no subject means nothing to relight"
