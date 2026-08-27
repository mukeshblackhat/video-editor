"""
Phase 2b: MatAnyone Video Matting — Soft Alpha Matte Generation

Replaces SAM 2's binary segmentation masks with continuous alpha mattes
that preserve hair strands, semi-transparent edges, and fine detail.

Pipeline integration:
  - Uses SAM 2 (phase2) to generate a FIRST-FRAME mask only
  - MatAnyone propagates that mask across all frames as soft alpha [0-255]
  - Output masks are drop-in replacements for phase2's binary masks

The key difference: segmentation is classification (0 or 1 per pixel),
matting is regression (0.0 to 1.0 per pixel). Matting preserves fractional
transparency at hair, motion blur, and soft edges.
"""

import torch
import cv2
import numpy as np
import os
import argparse
from pathlib import Path


def get_device():
    """Select best available device for MatAnyone."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_matanyone_model(device=None, model_path="pq-yang/MatAnyone"):
    """Load MatAnyone model from HuggingFace or local path.

    Args:
        device: torch.device or None (auto-detect).
        model_path: HuggingFace model ID or local checkpoint path.

    Returns:
        Tuple of (InferenceCore, device).
    """
    from matanyone.utils.get_default_model import get_matanyone_model
    from matanyone.inference.inference_core import InferenceCore
    from hydra.core.global_hydra import GlobalHydra

    if device is None:
        device = get_device()

    print(f"Loading MatAnyone model on {device}...")

    # Clear Hydra global state (may conflict with SAM 2 or prior initialization)
    GlobalHydra.instance().clear()

    # get_matanyone_model handles Hydra config initialization and weight loading
    network = get_matanyone_model(ckpt_path=model_path, device=device)
    core = InferenceCore(network, cfg=network.cfg, device=device)

    print("MatAnyone model loaded.")
    return core, device


def generate_first_frame_mask_sam2(frames_dir: str, prompt: dict,
                                   checkpoint_path: str, model_cfg: str,
                                   max_long_side: int = 1024) -> np.ndarray:
    """Use SAM 2 to generate a single first-frame mask for MatAnyone input.

    Args:
        frames_dir: Directory containing extracted frames (frame_NNNNNN.png).
        prompt: Dict with 'points' and 'labels' for SAM 2 prompting.
        checkpoint_path: Path to SAM 2 checkpoint.
        model_cfg: SAM 2 model config path.
        max_long_side: Max dimension for SAM 2 processing.

    Returns:
        Binary mask as uint8 ndarray (H, W), values 0 or 255, at original resolution.
    """
    from sam2.build_sam import build_sam2_video_predictor

    frames_dir = Path(frames_dir)
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        raise ValueError(f"No frames found in {frames_dir}")

    # Read first frame for dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        raise ValueError(f"Cannot read first frame: {frame_files[0]}")
    orig_h, orig_w = first_frame.shape[:2]

    # Downscale first frame for SAM 2
    scale = min(max_long_side / max(orig_h, orig_w), 1.0)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Create temp directory with just the first frame (downscaled)
    import tempfile
    import shutil
    tmp_dir = tempfile.mkdtemp(prefix="sam2_first_frame_")
    try:
        resized = cv2.resize(first_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(tmp_dir, "000000.jpg"), resized)

        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")

        predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=device)
        state = predictor.init_state(
            video_path=tmp_dir,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )

        # Scale prompt points
        points = np.array(prompt["points"], dtype=np.float32) * scale
        labels = np.array(prompt["labels"], dtype=np.int32)

        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
        )

        # Get mask for frame 0
        mask = None
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            if frame_idx == 0:
                mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(np.uint8) * 255
                break

        if mask is None:
            raise RuntimeError("SAM 2 did not produce a mask for frame 0")

        # Upscale to original resolution
        mask_full = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        # Keep as binary for MatAnyone input
        mask_full = (mask_full > 127).astype(np.uint8) * 255

        predictor.reset_state(state)
        del predictor
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

        return mask_full
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_matanyone_matting(frames_dir: str, first_frame_mask: np.ndarray,
                          masks_dir: str, n_warmup: int = 10,
                          r_erode: int = 10, r_dilate: int = 10,
                          max_size: int = 1024):
    """Run MatAnyone video matting on extracted frames.

    Uses the high-level process_video() API which handles all tensor
    formatting, warmup, and temporal propagation internally.

    Args:
        frames_dir: Directory of extracted frames (frame_NNNNNN.png).
        first_frame_mask: Binary mask uint8 (H, W) for the first frame.
        masks_dir: Output directory for alpha mattes.
        n_warmup: Number of warmup iterations on the first frame.
        r_erode: Erosion kernel radius for mask preprocessing.
        r_dilate: Dilation kernel radius for mask preprocessing.
        max_size: Max frame dimension for processing (1024 default).

    Returns:
        Total number of alpha mattes generated.
    """
    import torch.nn.functional as F
    from matanyone.utils.get_default_model import get_matanyone_model
    from matanyone.inference.inference_core import InferenceCore
    from matanyone.utils.inference_utils import gen_dilate, gen_erosion
    from hydra.core.global_hydra import GlobalHydra

    frames_dir = Path(frames_dir)
    masks_dir = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        raise ValueError(f"No frames found in {frames_dir}")

    total_frames = len(frame_files)
    device = get_device()

    # Read first frame for dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        raise ValueError(f"Cannot read frame: {frame_files[0]}")
    orig_h, orig_w = first_frame.shape[:2]

    # Clear Hydra global state (may be set by SAM 2 or prior model loading)
    GlobalHydra.instance().clear()

    print(f"Loading MatAnyone on {device}...")
    network = get_matanyone_model(
        ckpt_path="checkpoints/matanyone.pth", device=device)
    core = InferenceCore(network, cfg=network.cfg, device=device)

    # Preprocess mask using MatAnyone's own dilate/erode utils
    mask_np = first_frame_mask.copy()
    if r_dilate > 0:
        mask_np = gen_dilate(mask_np, r_dilate, r_dilate)
    if r_erode > 0:
        mask_np = gen_erosion(mask_np, r_erode, r_erode)

    # Determine processing size
    proc_h, proc_w = orig_h, orig_w
    need_resize = max_size > 0 and min(orig_h, orig_w) > max_size
    if need_resize:
        scale = max_size / min(orig_h, orig_w)
        proc_h = int(orig_h * scale)
        proc_w = int(orig_w * scale)
        print(f"  Processing at {proc_w}x{proc_h} (downscaled from {orig_w}x{orig_h})")

    # Prepare mask tensor at processing resolution
    mask_tensor = torch.from_numpy(mask_np).float().to(device)
    if need_resize:
        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0),
            size=(proc_h, proc_w), mode="nearest"
        )[0, 0]

    def load_frame(idx):
        """Load frame as (3, H, W) float tensor in [0, 255] range."""
        frame_bgr = cv2.imread(str(frame_files[idx]))
        if frame_bgr is None:
            raise ValueError(f"Cannot read frame: {frame_files[idx]}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1)
        if need_resize:
            tensor = F.interpolate(
                tensor.unsqueeze(0), size=(proc_h, proc_w), mode="area"
            )[0]
        return tensor

    total_iters = n_warmup + total_frames
    print(f"Processing {total_frames} frames with MatAnyone "
          f"(warmup={n_warmup}, erode={r_erode}, dilate={r_dilate}, "
          f"max_size={max_size})...")

    first_tensor = load_frame(0)

    with torch.inference_mode():
        for ti in range(total_iters):
            # Warmup: repeat first frame. After warmup: load actual frames.
            if ti < n_warmup:
                image = first_tensor.clone()
            else:
                frame_idx = ti - n_warmup
                image = first_tensor if frame_idx == 0 else load_frame(frame_idx)

            # Normalize to [0, 1]
            image_norm = (image / 255.0).float().to(device)

            # Exact same logic as MatAnyone's process_video (lines 513-520)
            if ti == 0:
                output_prob = core.step(image_norm, mask_tensor, objects=[1])
                output_prob = core.step(image_norm, first_frame_pred=True)
            elif ti <= n_warmup:
                output_prob = core.step(image_norm, first_frame_pred=True)
            else:
                output_prob = core.step(image_norm)

            # Save only after warmup completes
            if ti >= n_warmup:
                frame_idx = ti - n_warmup
                alpha = core.output_prob_to_mask(output_prob)
                alpha_np = alpha.unsqueeze(2).cpu().numpy()  # (H, W, 1)
                alpha_uint8 = np.clip(np.round(alpha_np * 255), 0, 255).astype(np.uint8)
                alpha_uint8 = alpha_uint8.squeeze()  # (H, W)

                # Upscale to original resolution if downscaled
                if alpha_uint8.shape[:2] != (orig_h, orig_w):
                    alpha_uint8 = cv2.resize(alpha_uint8, (orig_w, orig_h),
                                              interpolation=cv2.INTER_LINEAR)

                out_path = masks_dir / f"mask_{frame_idx:06d}.png"
                cv2.imwrite(str(out_path), alpha_uint8)

                if (frame_idx + 1) % 30 == 0:
                    print(f"  Matted {frame_idx + 1}/{total_frames} frames...")

    # Cleanup
    core.clear_memory()
    del core, network
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    actual_masks = len(list(masks_dir.glob("mask_*.png")))
    print(f"MatAnyone matting complete: {actual_masks} alpha mattes saved to {masks_dir}")
    return actual_masks


def run_full_matting_pipeline(frames_dir: str, prompt: dict,
                               checkpoint_path: str, model_cfg: str,
                               masks_dir: str, n_warmup: int = 10,
                               r_erode: int = 10, r_dilate: int = 10,
                               max_long_side: int = 1024, max_size: int = 1024):
    """Complete matting pipeline: SAM 2 first-frame → MatAnyone all-frames.

    Args:
        frames_dir: Directory of extracted frames.
        prompt: SAM 2 prompt dict with 'points' and 'labels'.
        checkpoint_path: SAM 2 checkpoint path.
        model_cfg: SAM 2 model config path.
        masks_dir: Output directory for alpha mattes.
        n_warmup: MatAnyone warmup iterations.
        r_erode: MatAnyone mask erosion radius.
        r_dilate: MatAnyone mask dilation radius.
        max_long_side: Max dimension for SAM 2 processing.
        max_size: Max dimension for MatAnyone processing (-1 = no limit).

    Returns:
        Total number of alpha mattes generated.
    """
    # Step 1: Generate first-frame mask using SAM 2
    print("Step 1: Generating first-frame mask with SAM 2...")
    first_frame_mask = generate_first_frame_mask_sam2(
        frames_dir, prompt, checkpoint_path, model_cfg, max_long_side
    )

    # Save the first-frame mask for reference
    masks_dir_path = Path(masks_dir)
    masks_dir_path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(masks_dir_path / "first_frame_mask_sam2.png"), first_frame_mask)
    print(f"  First-frame mask saved: {masks_dir_path / 'first_frame_mask_sam2.png'}")

    # Step 2: Run MatAnyone matting on all frames
    print("\nStep 2: Running MatAnyone video matting...")
    total = run_matanyone_matting(
        frames_dir, first_frame_mask, masks_dir,
        n_warmup=n_warmup, r_erode=r_erode, r_dilate=r_dilate,
        max_size=max_size,
    )

    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2b: MatAnyone video matting (soft alpha)")
    parser.add_argument("--frames-dir", default="outputs/frames",
                        help="Directory of extracted frames")
    parser.add_argument("--masks-dir", default="outputs/masks",
                        help="Output directory for alpha mattes")
    parser.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_large.pt",
                        help="SAM 2 checkpoint (for first-frame mask)")
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml",
                        help="SAM 2 model config")
    parser.add_argument("--points", required=True,
                        help="Foreground points at ORIGINAL resolution: 'x1,y1,x2,y2,...'")
    parser.add_argument("--n-warmup", type=int, default=10,
                        help="MatAnyone warmup iterations (default 10)")
    parser.add_argument("--r-erode", type=int, default=10,
                        help="Mask erosion radius (default 10)")
    parser.add_argument("--r-dilate", type=int, default=10,
                        help="Mask dilation radius (default 10)")
    parser.add_argument("--max-long-side", type=int, default=1024,
                        help="Max dimension for SAM 2 processing")
    parser.add_argument("--max-size", type=int, default=-1,
                        help="Max frame dimension for MatAnyone (-1 = no limit)")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    coords = list(map(int, args.points.split(",")))
    points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    prompt = {"points": points, "labels": [1] * len(points)}

    run_full_matting_pipeline(
        args.frames_dir, prompt, args.checkpoint, args.model_cfg,
        args.masks_dir, n_warmup=args.n_warmup, r_erode=args.r_erode,
        r_dilate=args.r_dilate, max_long_side=args.max_long_side,
        max_size=args.max_size,
    )
