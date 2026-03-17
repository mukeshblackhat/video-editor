"""
Phase 3: Composite — Replace Background
- Load original frames + masks + reference background image
- For each frame: foreground * mask + background * (1 - mask)
- Feather mask edges for natural blending
- Write composited frames
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path


def feather_mask(mask: np.ndarray, blur_radius: int = 5) -> np.ndarray:
    """Soften mask edges with Gaussian blur for natural blending."""
    if blur_radius <= 0:
        return mask.astype(np.float32) / 255.0

    # Ensure odd kernel size
    k = blur_radius * 2 + 1
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)
    return blurred / 255.0


def composite_frame(frame: np.ndarray, mask: np.ndarray,
                    background: np.ndarray, blur_radius: int = 5) -> np.ndarray:
    """Composite foreground onto new background using mask."""
    h, w = frame.shape[:2]

    # Resize background to match frame
    bg = cv2.resize(background, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Resize mask to match frame if needed
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    # Feather the mask
    alpha = feather_mask(mask, blur_radius)
    alpha = alpha[:, :, np.newaxis]  # (H, W, 1) for broadcasting

    # Composite: fg * alpha + bg * (1 - alpha)
    composited = (frame.astype(np.float32) * alpha +
                  bg.astype(np.float32) * (1 - alpha))

    return composited.astype(np.uint8)


def composite_all_frames(frames_dir: str, masks_dir: str, bg_image_path: str,
                         output_dir: str, blur_radius: int = 5):
    """Composite all frames with new background."""
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
    print(f"Compositing {count} frames with blur_radius={blur_radius}...")

    for i in range(count):
        frame = cv2.imread(str(frame_files[i]))
        mask = cv2.imread(str(mask_files[i]), cv2.IMREAD_GRAYSCALE)

        result = composite_frame(frame, mask, background, blur_radius)

        out_path = output_dir / f"comp_{i:06d}.png"
        cv2.imwrite(str(out_path), result)

        if (i + 1) % 30 == 0:
            print(f"  Composited {i + 1}/{count} frames...")

    print(f"Composited frames saved to {output_dir}")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Composite foreground onto new background")
    parser.add_argument("--frames-dir", default="outputs/frames", help="Original frames")
    parser.add_argument("--masks-dir", default="outputs/masks", help="Segmentation masks")
    parser.add_argument("--background", required=True, help="Path to new background image")
    parser.add_argument("--output-dir", default="outputs/composited", help="Output composited frames")
    parser.add_argument("--blur-radius", type=int, default=5, help="Mask edge feathering (0=hard)")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)
    composite_all_frames(args.frames_dir, args.masks_dir, args.background,
                         args.output_dir, args.blur_radius)
