"""
Full pipeline run: All 911 frames of test_full.mov with MatAnyone + all enhancements.
"""

import cv2
import os
from pathlib import Path

os.chdir(Path(__file__).parent)

VIDEO = "/Users/mukeshbishnoi/Desktop/test_full.mov"
BACKGROUND = "/Users/mukeshbishnoi/Desktop/frame.png"
POINTS = "1080,1800"

# Phase 1: Extract ALL frames
print("=" * 60)
print("PHASE 1: Extracting ALL frames")
print("=" * 60)

from phase1_extract import extract_frames
meta = extract_frames(VIDEO, "outputs/full_run/frames")

# Phase 2: MatAnyone matting
print("\n" + "=" * 60)
print("PHASE 2: SAM 2 first-frame + MatAnyone video matting")
print("=" * 60)

coords = list(map(int, POINTS.split(",")))
point_list = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
prompt = {"points": point_list, "labels": [1] * len(point_list)}

from phase2b_matting import generate_first_frame_mask_sam2, run_matanyone_matting

print("Generating first-frame mask with SAM 2...")
first_mask = generate_first_frame_mask_sam2(
    "outputs/full_run/frames", prompt,
    "checkpoints/sam2.1_hiera_large.pt",
    "configs/sam2.1/sam2.1_hiera_l.yaml",
    max_long_side=1024
)
os.makedirs("outputs/full_run/masks", exist_ok=True)
cv2.imwrite("outputs/full_run/first_frame_mask_sam2.png", first_mask)
print(f"SAM 2 mask: fg pixels = {(first_mask > 127).sum()}")

print("\nRunning MatAnyone matting on ALL frames...")
run_matanyone_matting(
    "outputs/full_run/frames", first_mask, "outputs/full_run/masks",
    n_warmup=10, r_erode=10, r_dilate=10, max_size=1024
)

# Phase 3: Composite
print("\n" + "=" * 60)
print("PHASE 3: Compositing ALL frames")
print("=" * 60)

from phase3_composite import composite_all_frames

composite_all_frames(
    "outputs/full_run/frames", "outputs/full_run/masks", BACKGROUND,
    "outputs/full_run/composited",
    blur_radius=5, edge_refine=True, edge_width=15,
    color_harmonize=True, harmonize_strength=0.6,
    decontaminate=True, decontaminate_strength=0.7,
    bg_motion=True,
)

# Phase 4: Render
print("\n" + "=" * 60)
print("PHASE 4: Rendering final video")
print("=" * 60)

from phase4_render import render_video, add_audio

render_video("outputs/full_run/composited", "outputs/full_run/final_no_audio.mp4", meta["fps"])
add_audio("outputs/full_run/final_no_audio.mp4", VIDEO, "outputs/full_run/final.mp4")

print("\n" + "=" * 60)
print(f"DONE! Output: {Path('outputs/full_run/final.mp4').resolve()}")
print("=" * 60)
