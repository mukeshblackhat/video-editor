"""
Side-by-side comparison: MatAnyone 1 vs MatAnyone 2.

Runs MatAnyone 2 over a set of frames using the SAME first-frame seed mask
that MatAnyone 1 was given, so the only variable is the matting model itself.
Reports timing and matte quality against the existing v1 mattes.

Run this with the MatAnyone 2 venv:

    venv_ma2/bin/python compare_matanyone.py \
        --frames outputs/frames_manvi --ref-masks outputs/masks_manvi \
        --out outputs/masks_ma2 --limit 40

Quality metrics mirror the ones used throughout this project:
  coverage%  - fraction of frame matted as subject
  soft%      - fraction with 0<alpha<255, i.e. the hair/edge detail
  hairMAE    - mean alpha error restricted to v1's soft band, where detail lives
  IoU        - silhouette agreement
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np


def load_seed_mask(ref_masks_dir: Path) -> np.ndarray:
    """Reuse the SAM 2 seed mask from the v1 run so both models start identical."""
    seed = ref_masks_dir / "first_frame_mask_sam2.png"
    if seed.exists():
        return cv2.imread(str(seed), cv2.IMREAD_GRAYSCALE)
    # Fall back to v1's own first output mask, binarised.
    first = sorted(ref_masks_dir.glob("mask_*.png"))
    if not first:
        raise ValueError(f"no masks to seed from in {ref_masks_dir}")
    m = cv2.imread(str(first[0]), cv2.IMREAD_GRAYSCALE)
    return (m > 127).astype(np.uint8) * 255


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_matanyone2(frames_dir: Path, seed_mask: np.ndarray, out_dir: Path,
                   limit: int, max_size: int):
    """Run MatAnyone 2 and return (seconds, frames_written)."""
    import torch
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(frames_dir.glob("frame_*.png"))[:limit]
    if not frame_files:
        raise ValueError(f"no frames in {frames_dir}")

    device = pick_device()
    print(f"device: {device}")

    from matanyone2 import MatAnyone2, InferenceCore

    model = MatAnyone2.from_pretrained("PeiqingYang/MatAnyone2")
    try:
        model = model.to(device)
    except Exception as exc:                     # device may be unsupported
        print(f"  could not move model to {device}: {exc}; falling back to cpu")
        device = "cpu"
        model = model.to(device)
    core = InferenceCore(model, device=device)

    # Write frames + seed to a temp video-like folder if the API needs paths.
    t0 = time.time()
    written = 0
    try:
        written = _run_stepwise(core, frame_files, seed_mask, out_dir,
                                device, max_size)
    except Exception as exc:
        print(f"  step-wise API unavailable ({exc}); trying process_video()")
        written = _run_process_video(core, frames_dir, seed_mask, out_dir, limit)
    return time.time() - t0, written


def _run_stepwise(core, frame_files, seed_mask, out_dir, device, max_size):
    """Preferred path: per-frame stepping, mirroring the v1 integration."""
    import torch
    import torch.nn.functional as F

    first = cv2.imread(str(frame_files[0]))
    orig_h, orig_w = first.shape[:2]

    proc_h, proc_w = orig_h, orig_w
    if max_size > 0 and min(orig_h, orig_w) > max_size:
        scale = max_size / min(orig_h, orig_w)
        proc_h, proc_w = int(orig_h * scale), int(orig_w * scale)
        print(f"  processing at {proc_w}x{proc_h}")

    mask_t = torch.from_numpy(seed_mask).float().to(device)
    if (proc_h, proc_w) != (orig_h, orig_w):
        mask_t = F.interpolate(mask_t[None, None], size=(proc_h, proc_w),
                               mode="nearest")[0, 0]

    def load(i):
        bgr = cv2.imread(str(frame_files[i]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).float().permute(2, 0, 1)
        if (proc_h, proc_w) != (orig_h, orig_w):
            t = F.interpolate(t[None], size=(proc_h, proc_w), mode="area")[0]
        return t

    n_warmup = 10
    first_t = load(0)
    written = 0
    with torch.inference_mode():
        for ti in range(n_warmup + len(frame_files)):
            img = first_t.clone() if ti < n_warmup else (
                first_t if ti - n_warmup == 0 else load(ti - n_warmup))
            img = (img / 255.0).float().to(device)

            if ti == 0:
                core.step(img, mask_t, objects=[1])
                prob = core.step(img, first_frame_pred=True)
            elif ti <= n_warmup:
                prob = core.step(img, first_frame_pred=True)
            else:
                prob = core.step(img)

            if ti >= n_warmup:
                idx = ti - n_warmup
                alpha = core.output_prob_to_mask(prob)
                a = alpha.unsqueeze(2).cpu().numpy().squeeze()
                a = np.clip(np.round(a * 255), 0, 255).astype(np.uint8)
                if a.shape[:2] != (orig_h, orig_w):
                    a = cv2.resize(a, (orig_w, orig_h),
                                   interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(out_dir / f"mask_{idx:06d}.png"), a)
                written += 1
    return written


def _run_process_video(core, frames_dir, seed_mask, out_dir, limit):
    """Fallback: the documented high-level API, which wants file paths."""
    import tempfile
    import shutil
    tmp = Path(tempfile.mkdtemp(prefix="ma2_"))
    try:
        seed_path = tmp / "seed.png"
        cv2.imwrite(str(seed_path), seed_mask)
        vid = tmp / "clip.mp4"
        files = sorted(frames_dir.glob("frame_*.png"))[:limit]
        h, w = cv2.imread(str(files[0])).shape[:2]
        vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))
        for f in files:
            vw.write(cv2.imread(str(f)))
        vw.release()
        core.process_video(str(vid), str(seed_path), str(out_dir))
        return len(list(out_dir.glob("*.png")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def compare(ref_dir: Path, new_dir: Path, limit: int):
    """Report v1 vs v2 matte quality on the same frames."""
    print(f"\n{'frame':>6}{'v1 cov%':>9}{'v2 cov%':>9}{'v1 soft%':>10}"
          f"{'v2 soft%':>10}{'hairMAE':>9}{'IoU%':>8}")
    rows = []
    for i in range(0, min(limit, 40), 8):
        r = cv2.imread(str(ref_dir / f"mask_{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        n = cv2.imread(str(new_dir / f"mask_{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if r is None or n is None:
            continue
        if n.shape != r.shape:
            n = cv2.resize(n, (r.shape[1], r.shape[0]))
        rf, nf = r.astype(np.float32), n.astype(np.float32)
        band = (r > 10) & (r < 245)
        hair = np.abs(nf - rf)[band].mean() if band.any() else 0.0
        a, b = n > 127, r > 127
        iou = (a & b).sum() / max((a | b).sum(), 1) * 100
        row = ((r > 127).mean() * 100, (n > 127).mean() * 100,
               band.mean() * 100, (((n > 10) & (n < 245)).mean()) * 100,
               hair, iou)
        rows.append(row)
        print(f"{i:6d}{row[0]:9.1f}{row[1]:9.1f}{row[2]:10.2f}"
              f"{row[3]:10.2f}{row[4]:9.1f}{row[5]:8.1f}")
    if rows:
        m = np.array(rows).mean(axis=0)
        print(f"{'MEAN':>6}{m[0]:9.1f}{m[1]:9.1f}{m[2]:10.2f}"
              f"{m[3]:10.2f}{m[4]:9.1f}{m[5]:8.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare MatAnyone 1 vs 2")
    ap.add_argument("--frames", default="outputs/frames_manvi")
    ap.add_argument("--ref-masks", default="outputs/masks_manvi")
    ap.add_argument("--out", default="outputs/masks_ma2")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--max-size", type=int, default=768)
    args = ap.parse_args()

    os.chdir(Path(__file__).parent)
    frames, ref, out = Path(args.frames), Path(args.ref_masks), Path(args.out)

    seed = load_seed_mask(ref)
    print(f"seed mask: {seed.shape}, coverage {(seed > 127).mean() * 100:.1f}%")

    secs, n = run_matanyone2(frames, seed, out, args.limit, args.max_size)
    print(f"\nMatAnyone 2: {secs:.1f}s for {n} frames = {secs / max(n, 1):.3f}s/frame")
    print(f"MatAnyone 1 reference: 0.340s/frame (122s / 359 frames at 768)")

    compare(ref, out, args.limit)
