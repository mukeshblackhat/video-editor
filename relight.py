"""
Subject relighting for compositing.

A composited subject reads as a cutout when its lighting disagrees with the
background it was placed into. Three things give it away, and this module
addresses each:

  1. **Exposure** - the subject was lit for its original room. Dropped into a
     darker or brighter scene it floats, because its overall luminance does
     not belong.
  2. **Ambient bounce** - real rooms bounce colour onto people. A subject in a
     purple room picks up purple; one in a white studio picks up neutral fill.
  3. **Directional key** - light in the new scene comes from somewhere. The
     sides of the subject facing that direction should be brighter, the
     opposite sides darker.

A fourth function, ``add_contact_shadow``, grounds the subject by darkening
the background where it meets them.

Unlike ``color_harmonize``, which works on global statistics and edge pixels,
these operate on the subject's **interior**. That is the difference between
tinting a silhouette and lighting a person.

This is a 2D approximation. Without a depth or normal map, surface
orientation is inferred from mask geometry and luminance, so it will not match
true 3D relighting - but it removes the cues that read as "pasted on".

Dependencies: cv2, numpy.
"""

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Light analysis
# --------------------------------------------------------------------------- #

def estimate_light_direction(bg: np.ndarray) -> tuple:
    """Infer where the background's key light comes from.

    Compares luminance across the horizontal and vertical halves of the
    background. The result points *toward* the light source.

    Args:
        bg: Background image, uint8 (H, W, 3) BGR.

    Returns:
        (dx, dy) unit-ish vector. Negative dx means light from the left,
        negative dy means light from above. Magnitude reflects how strongly
        directional the lighting is; a flat background returns near (0, 0).
    """
    if bg is None or bg.size == 0:
        raise ValueError("bg must be a non-empty array")

    lum = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = lum.shape

    left, right = lum[:, :w // 2].mean(), lum[:, w // 2:].mean()
    top, bottom = lum[:h // 2].mean(), lum[h // 2:].mean()

    # Normalise the imbalance by the overall level so the vector is scale-free.
    scale = max(lum.mean(), 1.0)
    dx = (right - left) / scale
    dy = (bottom - top) / scale

    mag = np.hypot(dx, dy)
    if mag > 1.0:                       # keep it within the unit circle
        dx, dy = dx / mag, dy / mag
    return float(dx), float(dy)


# --------------------------------------------------------------------------- #
# 1. Exposure
# --------------------------------------------------------------------------- #

def match_subject_exposure(frame: np.ndarray, alpha: np.ndarray,
                           bg: np.ndarray, strength: float = 0.5,
                           preserve_contrast: float = 0.7) -> np.ndarray:
    """Move the subject's overall brightness toward the background's.

    Applies a gain in LAB lightness rather than a flat offset, so shadows and
    highlights scale proportionally and internal contrast survives.
    ``preserve_contrast`` pulls the gain back toward 1.0 to avoid flattening
    the subject into a silhouette.

    Args:
        frame: Video frame, uint8 (H, W, 3) BGR.
        alpha: Soft alpha matte, float32 (H, W) in [0, 1].
        bg: Background already fitted to frame size, uint8 (H, W, 3) BGR.
        strength: How far to close the gap, [0, 1].
        preserve_contrast: How much of the original contrast to keep, [0, 1].

    Returns:
        Relit frame, uint8 (H, W, 3) BGR. Input is not modified.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength == 0.0:
        return frame.copy()

    subject = alpha > 0.5
    if not np.any(subject):
        return frame.copy()

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float32)

    subj_l = lab[:, :, 0][subject].mean()
    bg_l = bg_lab[:, :, 0].mean()
    if subj_l < 1.0:
        return frame.copy()

    # Multiplicative gain toward the background level, damped by strength.
    target = subj_l + (bg_l - subj_l) * strength
    gain = target / subj_l
    # Pull back toward 1.0 so we shift exposure without crushing contrast.
    gain = 1.0 + (gain - 1.0) * (1.0 - preserve_contrast * 0.5)

    # Weight by alpha so the effect fades out across the matte edge.
    w = np.clip(alpha, 0.0, 1.0)[:, :, None]
    lit = lab.copy()
    lit[:, :, 0] = lab[:, :, 0] * gain
    out_lab = lab * (1.0 - w) + lit * w
    out_lab[:, :, 0] = np.clip(out_lab[:, :, 0], 0, 255)

    return cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8),
                        cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------- #
# 2. Ambient bounce
# --------------------------------------------------------------------------- #

def apply_ambient_bounce(frame: np.ndarray, alpha: np.ndarray,
                         bg: np.ndarray, strength: float = 0.25) -> np.ndarray:
    """Tint the subject toward the background's dominant colour.

    Rooms bounce light onto the people in them. This applies that cast across
    the subject's whole body - weighted slightly toward darker regions, which
    is where bounce light is most visible in practice, since bright areas are
    already dominated by the key light.

    Args:
        frame: Video frame, uint8 (H, W, 3) BGR.
        alpha: Soft alpha matte, float32 (H, W) in [0, 1].
        bg: Background fitted to frame size, uint8 (H, W, 3) BGR.
        strength: Bounce intensity, [0, 1].

    Returns:
        Tinted frame, uint8 (H, W, 3) BGR. Input is not modified.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength == 0.0:
        return frame.copy()

    subject = alpha > 0.01
    if not np.any(subject):
        return frame.copy()

    bg_color = bg.reshape(-1, 3).astype(np.float32).mean(axis=0)   # BGR

    f = frame.astype(np.float32)
    lum = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Shadow regions receive proportionally more bounce than lit ones.
    shadow_weight = (1.0 - lum) * 0.6 + 0.4
    w = (np.clip(alpha, 0.0, 1.0) * shadow_weight * strength)[:, :, None]

    out = f * (1.0 - w) + bg_color[None, None, :] * w
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 3. Directional key
# --------------------------------------------------------------------------- #

def apply_directional_light(frame: np.ndarray, alpha: np.ndarray,
                            direction: tuple, strength: float = 0.3,
                            falloff: float = 1.0) -> np.ndarray:
    """Brighten the side of the subject facing the light, darken the other.

    Surface orientation is approximated from the subject's horizontal and
    vertical extent: pixels toward the light side of the subject's centroid
    are lit, pixels away from it fall into shadow. This is the cue that makes
    a composite read as belonging to the scene.

    Args:
        frame: Video frame, uint8 (H, W, 3) BGR.
        alpha: Soft alpha matte, float32 (H, W) in [0, 1].
        direction: (dx, dy) pointing toward the light, as returned by
            :func:`estimate_light_direction`. dx < 0 means the light is on
            the left, dy < 0 means it is above.
        strength: Lighting intensity, [0, 1].
        falloff: Exponent on the gradient; higher concentrates the effect.

    Returns:
        Relit frame, uint8 (H, W, 3) BGR. Input is not modified.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength == 0.0:
        return frame.copy()

    subject = alpha > 0.5
    if not np.any(subject):
        return frame.copy()

    dx, dy = direction
    mag = np.hypot(dx, dy)
    if mag < 1e-6:
        return frame.copy()
    dx, dy = dx / mag, dy / mag

    h, w = alpha.shape
    ys, xs = np.where(subject)
    cx, cy = xs.mean(), ys.mean()
    span_x = max(xs.max() - xs.min(), 1)
    span_y = max(ys.max() - ys.min(), 1)

    # Signed position along the light axis, normalised to roughly [-1, 1].
    gx = (np.arange(w, dtype=np.float32)[None, :] - cx) / (span_x * 0.5)
    gy = (np.arange(h, dtype=np.float32)[:, None] - cy) / (span_y * 0.5)
    # `direction` points toward the light (dx<0 means the light is on the
    # left), so pixels whose offset agrees with it are the lit ones.
    grad = np.clip(gx * dx + gy * dy, -1.0, 1.0)
    grad = np.sign(grad) * (np.abs(grad) ** falloff)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    # +-28 L* at full strength: a visible key without blowing out skin.
    delta = grad * 28.0 * strength * np.clip(alpha, 0.0, 1.0)
    lab[:, :, 0] = np.clip(lab[:, :, 0] + delta, 0, 255)

    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------- #
# 4. Contact shadow
# --------------------------------------------------------------------------- #

def add_contact_shadow(bg: np.ndarray, alpha: np.ndarray,
                       strength: float = 0.4, offset: tuple = (12, 18),
                       blur: int = 41) -> np.ndarray:
    """Darken the background where the subject meets it.

    Without a cast shadow a subject floats regardless of how well its colour
    matches. This offsets and blurs the matte, then multiplies it into the
    background as a soft occlusion.

    Args:
        bg: Background fitted to frame size, uint8 (H, W, 3) BGR.
        alpha: Soft alpha matte, float32 (H, W) in [0, 1].
        strength: Shadow darkness, [0, 1].
        offset: (dx, dy) shadow displacement in pixels.
        blur: Gaussian blur kernel size; forced odd.

    Returns:
        Background with the shadow composited in, uint8 (H, W, 3) BGR.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength == 0.0:
        return bg.copy()
    if not np.any(alpha > 0.01):
        return bg.copy()

    h, w = alpha.shape
    dx, dy = int(offset[0]), int(offset[1])

    shadow = np.zeros_like(alpha, dtype=np.float32)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    shadow[ys0:ys1, xs0:xs1] = alpha[sy0:sy1, sx0:sx1]

    k = blur if blur % 2 == 1 else blur + 1
    shadow = cv2.GaussianBlur(shadow, (k, k), 0)

    # The subject itself will be drawn on top, so keep the shadow outside it.
    shadow = np.clip(shadow - alpha, 0.0, 1.0) * strength

    out = bg.astype(np.float32) * (1.0 - shadow[:, :, None] * 0.65)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def relight_subject(frame: np.ndarray, alpha: np.ndarray, bg: np.ndarray,
                    exposure: float = 0.5, bounce: float = 0.25,
                    directional: float = 0.3,
                    light_direction: tuple = None) -> np.ndarray:
    """Apply the full relighting chain to a subject.

    Order matters: exposure sets the overall level, bounce adds the room's
    colour, and the directional key is applied last so it shapes the
    already-corrected image.

    Args:
        frame: Video frame, uint8 (H, W, 3) BGR.
        alpha: Soft alpha matte, float32 (H, W) in [0, 1].
        bg: Background fitted to frame size, uint8 (H, W, 3) BGR.
        exposure: Exposure-match strength, [0, 1]. 0 disables.
        bounce: Ambient bounce strength, [0, 1]. 0 disables.
        directional: Directional key strength, [0, 1]. 0 disables.
        light_direction: (dx, dy) toward the light. None estimates it from bg.

    Returns:
        Relit frame, uint8 (H, W, 3) BGR. Input is not modified.

    Raises:
        ValueError: If frame, alpha, and bg shapes disagree.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must be (H,W,3), got {frame.shape}")
    if alpha.shape[:2] != frame.shape[:2]:
        raise ValueError(
            f"alpha shape {alpha.shape} does not match frame {frame.shape[:2]}")
    if bg.shape[:2] != frame.shape[:2]:
        raise ValueError(
            f"bg shape {bg.shape[:2]} does not match frame {frame.shape[:2]}")

    if not np.any(alpha > 0.01):
        return frame.copy()

    out = frame
    if exposure > 0.0:
        out = match_subject_exposure(out, alpha, bg, strength=exposure)
    if bounce > 0.0:
        out = apply_ambient_bounce(out, alpha, bg, strength=bounce)
    if directional > 0.0:
        d = light_direction if light_direction is not None else \
            estimate_light_direction(bg)
        out = apply_directional_light(out, alpha, d, strength=directional)

    return out if out is not frame else frame.copy()
