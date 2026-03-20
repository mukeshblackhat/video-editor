"""
Phase 2: SAM 2 Video Segmentation (Chunked)
- Downscale frames for processing
- Split into chunks to avoid MPS memory limits
- Run SAM 2 on each chunk with prompt propagation
- Upscale masks back to original resolution
"""

import torch
import numpy as np
import cv2
import os
import shutil
import argparse
import json
from pathlib import Path


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def prepare_downscaled_frames(frames_dir: str, scaled_dir: str, max_long_side: int = 1024):
    """Downscale frames for SAM 2 processing. Returns scale info."""
    frames_dir = Path(frames_dir)
    scaled_dir = Path(scaled_dir)
    scaled_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(frames_dir.glob("frame_*.png"))
    first = cv2.imread(str(frame_files[0]))
    orig_h, orig_w = first.shape[:2]

    scale = min(max_long_side / max(orig_h, orig_w), 1.0)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    print(f"Downscaling: {orig_w}x{orig_h} -> {new_w}x{new_h} (scale={scale:.3f})")

    for i, f in enumerate(frame_files):
        frame = cv2.imread(str(f))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        out_path = scaled_dir / f"{i:06d}.jpg"
        cv2.imwrite(str(out_path), resized)
        if (i + 1) % 100 == 0:
            print(f"  Downscaled {i + 1}/{len(frame_files)} frames...")

    print(f"Downscaled {len(frame_files)} frames to {scaled_dir}")
    return {
        "scale": scale,
        "orig_w": orig_w,
        "orig_h": orig_h,
        "new_w": new_w,
        "new_h": new_h,
        "total_frames": len(frame_files),
    }


def split_into_chunk_dirs(scaled_dir: str, chunk_size: int):
    """Split scaled frames into chunk subdirectories. Returns list of (chunk_dir, start_idx)."""
    scaled_dir = Path(scaled_dir)
    all_files = sorted(scaled_dir.glob("*.jpg"))
    total = len(all_files)

    chunks = []
    for chunk_start in range(0, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        chunk_dir = scaled_dir / f"chunk_{chunk_start:06d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        for local_idx, global_idx in enumerate(range(chunk_start, chunk_end)):
            src = all_files[global_idx]
            dst = chunk_dir / f"{local_idx:06d}.jpg"
            shutil.copy2(str(src), str(dst))

        chunks.append((str(chunk_dir), chunk_start, chunk_end - chunk_start))
        print(f"  Chunk: frames {chunk_start}-{chunk_end - 1} ({chunk_end - chunk_start} frames)")

    return chunks


def segment_chunk(chunk_dir: str, chunk_start: int, chunk_len: int,
                  prompt: dict, checkpoint_path: str, model_cfg: str,
                  masks_dir: str, scale_info: dict, device: str):
    """Run SAM 2 on a single chunk of frames."""
    from sam2.build_sam import build_sam2_video_predictor

    masks_dir = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=device)

    state = predictor.init_state(
        video_path=chunk_dir,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )

    # Scale prompt points to match downscaled resolution
    scale = scale_info["scale"]
    points = np.array(prompt["points"], dtype=np.float32) * scale
    labels = np.array(prompt["labels"], dtype=np.int32)

    # Always prompt on frame 0 of this chunk
    _, obj_ids, _ = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        obj_id=1,
        points=points,
        labels=labels,
    )

    orig_w = scale_info["orig_w"]
    orig_h = scale_info["orig_h"]

    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(np.uint8) * 255

        # Upscale mask to original resolution — keep soft edges from
        # bilinear interpolation instead of re-binarizing.
        mask_full = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        global_idx = chunk_start + frame_idx
        mask_path = masks_dir / f"mask_{global_idx:06d}.png"
        cv2.imwrite(str(mask_path), mask_full)

    predictor.reset_state(state)
    # Free GPU memory
    del predictor
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def run_segmentation(frames_dir: str, prompt: dict, checkpoint_path: str,
                     model_cfg: str, masks_dir: str, max_long_side: int = 1024,
                     chunk_size: int = 500):
    """Full segmentation pipeline with chunking."""
    device = get_device()
    print(f"Using device: {device}")

    # Step 1: Downscale
    scaled_dir = "outputs/frames_scaled"
    scale_info = prepare_downscaled_frames(frames_dir, scaled_dir, max_long_side)

    # Step 2: Split into chunks
    print(f"\nSplitting {scale_info['total_frames']} frames into chunks of {chunk_size}...")
    chunks = split_into_chunk_dirs(scaled_dir, chunk_size)

    # Step 3: Process each chunk
    for i, (chunk_dir, chunk_start, chunk_len) in enumerate(chunks):
        print(f"\n--- Processing chunk {i + 1}/{len(chunks)} "
              f"(frames {chunk_start}-{chunk_start + chunk_len - 1}) ---")
        segment_chunk(
            chunk_dir, chunk_start, chunk_len,
            prompt, checkpoint_path, model_cfg,
            masks_dir, scale_info, device,
        )
        print(f"  Chunk {i + 1} done.")

    total_masks = len(list(Path(masks_dir).glob("mask_*.png")))
    print(f"\nAll done! {total_masks} masks saved to {masks_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: SAM 2 video segmentation (chunked)")
    parser.add_argument("--frames-dir", default="outputs/frames", help="Original frames directory")
    parser.add_argument("--masks-dir", default="outputs/masks", help="Output masks directory")
    parser.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_large.pt",
                        help="SAM 2 checkpoint path")
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml",
                        help="SAM 2 model config")
    parser.add_argument("--points", required=True,
                        help="Comma-separated x,y points at ORIGINAL resolution e.g. '1080,2000'")
    parser.add_argument("--max-long-side", type=int, default=1024,
                        help="Max long side for downscaled frames")
    parser.add_argument("--chunk-size", type=int, default=450,
                        help="Frames per chunk (default 450)")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    coords = list(map(int, args.points.split(",")))
    points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    prompt = {"points": points, "labels": [1] * len(points)}

    run_segmentation(args.frames_dir, prompt, args.checkpoint, args.model_cfg,
                     args.masks_dir, args.max_long_side, args.chunk_size)
