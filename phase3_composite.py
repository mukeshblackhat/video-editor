"""
Phase 3: Composite — Replace Background (Enhanced)

Full compositing pipeline with:
  - Edge-aware mask refinement (guided filter, morphological cleanup)
  - Color spill decontamination (removes original background bleeding)
  - Lighting & color harmonization (histogram matching, white balance, exposure)
  - Background movement (Ken Burns, parallax, depth parallax, motion blur)
  - Proper float32 → uint8 rounding (no truncation)
  - Cached background resize (once, not per-frame)
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path

from edge_refine import refine_mask_edges, decontaminate_edges, suppress_edge_spill
from color_harmonize import harmonize_colors
from relight import relight_subject, add_contact_shadow
from bg_motion import (apply_background_motion, get_default_motion_config,
                       fit_background_to_frame, aspect_mismatch,
                       ASPECT_WARN_THRESHOLD)


def feather_mask(mask: np.ndarray, blur_radius: int = 5) -> np.ndarray:
    """Legacy simple feathering — kept for backward compatibility.

    Prefer refine_mask_edges() from edge_refine module for quality results.
    """
    if blur_radius <= 0:
        return mask.astype(np.float32) / 255.0
    k = blur_radius * 2 + 1
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)
    return blurred / 255.0


def composite_frame(frame: np.ndarray, mask: np.ndarray,
                    bg_resized: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Composite foreground onto background using pre-computed alpha.

    Args:
        frame: Original video frame, uint8 (H, W, 3) BGR.
        mask: Raw mask, uint8 (H, W) — used only if alpha is None.
        bg_resized: Background already resized to frame dims, uint8 (H, W, 3).
        alpha: Pre-computed soft alpha, float32 (H, W) in [0, 1].

    Returns:
        Composited frame, uint8 (H, W, 3) BGR.
    """
    alpha_3d = alpha[:, :, np.newaxis]  # (H, W, 1) for broadcasting

    composited = (frame.astype(np.float32) * alpha_3d +
                  bg_resized.astype(np.float32) * (1.0 - alpha_3d))

    # Round instead of truncate for accuracy
    return np.clip(np.round(composited), 0, 255).astype(np.uint8)


def composite_all_frames(frames_dir: str, masks_dir: str, bg_image_path: str,
                         output_dir: str, blur_radius: int = 5,
                         edge_refine: bool = True, edge_width: int = 15,
                         color_harmonize: bool = True, harmonize_strength: float = 0.6,
                         decontaminate: bool = True, decontaminate_strength: float = 0.7,
                         bg_motion: bool = True, motion_config: dict = None,
                         spill_suppress: bool = True, spill_strength: float = 1.0,
                         bg_fit: str = "cover", relight: bool = False,
                         relight_exposure: float = 0.25, relight_bounce: float = 0.15,
                         relight_directional: float = 0.35,
                         contact_shadow: float = 0.55):
    """Composite all frames with new background using the full enhancement pipeline.

    Args:
        frames_dir: Directory of original frames (frame_NNNNNN.png).
        masks_dir: Directory of segmentation masks (mask_NNNNNN.png).
        bg_image_path: Path to the new background image.
        output_dir: Output directory for composited frames.
        blur_radius: Edge feathering radius (used by both legacy and refined).
        edge_refine: Enable edge-aware mask refinement.
        edge_width: Width of the edge transition band in pixels.
        color_harmonize: Enable color/lighting harmonization.
        harmonize_strength: Harmonization strength [0, 1].
        decontaminate: Enable edge color decontamination.
        decontaminate_strength: Decontamination strength [0, 1].
        bg_motion: Enable background movement effects.
        motion_config: Background motion config dict. None = use defaults.
        spill_suppress: Remove original-background light from edge pixels.
        spill_strength: Spill suppression strength [0, 1].
        bg_fit: Background fit mode: 'cover', 'contain', or 'stretch'.
        relight: Relight the subject to match the background's lighting.
        relight_exposure: Exposure-match strength [0, 1].
        relight_bounce: Ambient bounce strength [0, 1].
        relight_directional: Directional key strength [0, 1].
        contact_shadow: Contact shadow strength [0, 1].

    Returns:
        Number of frames composited.
    """
    frames_dir = Path(frames_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    background = cv2.imread(bg_image_path)
    if background is None:
        raise ValueError(f"Cannot read background image: {bg_image_path}")

    frame_files = sorted(frames_dir.glob("frame_*.png"))
    mask_files = sorted(masks_dir.glob("mask_*.png"))

    if len(frame_files) != len(mask_files):
        print(f"Warning: {len(frame_files)} frames but {len(mask_files)} masks. "
              f"Processing min({len(frame_files)}, {len(mask_files)}) frames.")

    count = min(len(frame_files), len(mask_files))
    if count == 0:
        raise ValueError(f"No matching frames/masks found in {frames_dir} / {masks_dir}")

    if motion_config is None and bg_motion:
        motion_config = get_default_motion_config()

    # Pre-read first frame to get dimensions and cache resized background
    first_frame = cv2.imread(str(frame_files[0]))
    h, w = first_frame.shape[:2]

    bg_h, bg_w = background.shape[:2]
    mismatch = aspect_mismatch(bg_w, bg_h, w, h)
    if mismatch > ASPECT_WARN_THRESHOLD and bg_fit != "stretch":
        print(f"  Note: background {bg_w}x{bg_h} (AR {bg_w/bg_h:.3f}) differs from "
              f"frame {w}x{h} (AR {w/h:.3f}) by {mismatch*100:.1f}%; "
              f"using '{bg_fit}' fit.")

    # Fit once at frame size. Background motion then operates in the correct
    # aspect space instead of being stretched after the fact.
    background = fit_background_to_frame(background, w, h, mode=bg_fit)
    bg_resized_cache = background

    features = []
    if edge_refine:
        features.append("edge-refine")
    if color_harmonize:
        features.append("color-harmonize")
    if decontaminate:
        features.append("decontaminate")
    if spill_suppress:
        features.append("spill-suppress")
    if relight:
        features.append("relight")
    if relight and contact_shadow > 0:
        features.append("contact-shadow")
    if bg_motion:
        features.append("bg-motion")
    print(f"Compositing {count} frames | Features: {', '.join(features) if features else 'basic'}")

    for i in range(count):
        frame = cv2.imread(str(frame_files[i]))
        mask = cv2.imread(str(mask_files[i]), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise ValueError(f"Cannot read frame: {frame_files[i]}")
        if mask is None:
            raise ValueError(f"Cannot read mask: {mask_files[i]}")

        # Resize mask if needed
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        # Step 1: Refine mask edges (or use legacy feathering)
        # Detect if mask is already soft alpha (from MatAnyone) by checking
        # for values between 1 and 254. Binary masks have only 0 and 255.
        unique_vals = np.unique(mask)
        is_soft_alpha = np.any((unique_vals > 0) & (unique_vals < 255))

        if is_soft_alpha:
            # MatAnyone already produced soft alpha — use directly
            alpha = mask.astype(np.float32) / 255.0
        elif edge_refine:
            alpha = refine_mask_edges(mask, frame, blur_radius=blur_radius,
                                      edge_width=edge_width)
        else:
            alpha = feather_mask(mask, blur_radius)

        # Step 2a: Suppress spill from the ORIGINAL background. Must run
        # before compositing, while the old background colour is still
        # recoverable from the frame.
        if spill_suppress:
            frame = suppress_edge_spill(frame, alpha, strength=spill_strength)

        # Step 2b: Decontaminate edge colors
        if decontaminate:
            frame = decontaminate_edges(frame, alpha, strength=decontaminate_strength)

        # Step 3: Apply background motion
        if bg_motion and motion_config:
            bg_frame = apply_background_motion(
                background, i, count, mask, motion_config
            )
        else:
            bg_frame = bg_resized_cache

        # Step 4: Color harmonization
        if color_harmonize:
            frame = harmonize_colors(frame, bg_frame, alpha,
                                     strength=harmonize_strength)

        # Step 4b: Relight the subject to sit in the new scene's light, and
        # ground it with a contact shadow. Both need bg_frame, so they run
        # after background motion has been resolved.
        if relight:
            frame = relight_subject(
                frame, alpha, bg_frame,
                exposure=relight_exposure, bounce=relight_bounce,
                directional=relight_directional,
            )
            if contact_shadow > 0:
                bg_frame = add_contact_shadow(
                    bg_frame, alpha, strength=contact_shadow,
                    offset=(16, 26), blur=61,
                )

        # Step 5: Final composite
        result = composite_frame(frame, mask, bg_frame, alpha)

        out_path = output_dir / f"comp_{i:06d}.png"
        cv2.imwrite(str(out_path), result)

        if (i + 1) % 30 == 0:
            print(f"  Composited {i + 1}/{count} frames...")

    print(f"Composited frames saved to {output_dir}")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 3: Composite foreground onto new background (enhanced)")
    parser.add_argument("--frames-dir", default="outputs/frames", help="Original frames")
    parser.add_argument("--masks-dir", default="outputs/masks", help="Segmentation masks")
    parser.add_argument("--background", required=True, help="Path to new background image")
    parser.add_argument("--output-dir", default="outputs/composited", help="Output dir")
    parser.add_argument("--blur-radius", type=int, default=5, help="Mask edge feathering")
    parser.add_argument("--edge-width", type=int, default=15, help="Edge transition band width")
    parser.add_argument("--no-edge-refine", action="store_true",
                        help="Disable edge-aware refinement (use simple Gaussian)")
    parser.add_argument("--no-color-harmonize", action="store_true",
                        help="Disable color/lighting harmonization")
    parser.add_argument("--harmonize-strength", type=float, default=0.6,
                        help="Color harmonization strength [0-1]")
    parser.add_argument("--no-decontaminate", action="store_true",
                        help="Disable edge color decontamination")
    parser.add_argument("--decontaminate-strength", type=float, default=0.7,
                        help="Edge decontamination strength [0-1]")
    parser.add_argument("--no-bg-motion", action="store_true",
                        help="Disable background movement effects")
    parser.add_argument("--no-spill-suppress", action="store_true",
                        help="Disable original-background spill suppression")
    parser.add_argument("--spill-strength", type=float, default=1.0,
                        help="Spill suppression strength [0-1]")
    parser.add_argument("--relight", action="store_true",
                        help="Relight the subject to match background lighting")
    parser.add_argument("--relight-exposure", type=float, default=0.25,
                        help="Exposure-match strength [0-1]")
    parser.add_argument("--relight-bounce", type=float, default=0.15,
                        help="Ambient bounce strength [0-1]")
    parser.add_argument("--relight-directional", type=float, default=0.35,
                        help="Directional key strength [0-1]")
    parser.add_argument("--contact-shadow", type=float, default=0.55,
                        help="Contact shadow strength [0-1]")
    parser.add_argument("--bg-fit", choices=["cover", "contain", "stretch"],
                        default="cover",
                        help="How to fit a background whose aspect differs from the video")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    composite_all_frames(
        args.frames_dir, args.masks_dir, args.background, args.output_dir,
        blur_radius=args.blur_radius,
        edge_refine=not args.no_edge_refine,
        edge_width=args.edge_width,
        color_harmonize=not args.no_color_harmonize,
        harmonize_strength=args.harmonize_strength,
        decontaminate=not args.no_decontaminate,
        decontaminate_strength=args.decontaminate_strength,
        bg_motion=not args.no_bg_motion,
        spill_suppress=not args.no_spill_suppress,
        spill_strength=args.spill_strength,
        bg_fit=args.bg_fit,
        relight=args.relight,
        relight_exposure=args.relight_exposure,
        relight_bounce=args.relight_bounce,
        relight_directional=args.relight_directional,
        contact_shadow=args.contact_shadow,
    )
