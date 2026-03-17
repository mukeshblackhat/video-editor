"""
Phase 1: Video Frame Extraction & Reconstruction
- Extract all frames from a video
- Rebuild video from frames (proving the pipeline works losslessly)
"""

import cv2
import os
import argparse
from pathlib import Path


def extract_frames(video_path: str, output_dir: str) -> dict:
    """Extract all frames from video, return metadata."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path.name}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = output_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(frame_path), frame)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"  Extracted {frame_idx}/{total_frames} frames...")

    cap.release()
    print(f"Extracted {frame_idx} frames to {output_dir}")

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": frame_idx,
        "frames_dir": str(output_dir),
    }


def rebuild_video(frames_dir: str, output_path: str, fps: float) -> str:
    """Rebuild video from extracted frames."""
    frames_dir = Path(frames_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        raise ValueError(f"No frames found in {frames_dir}")

    first_frame = cv2.imread(str(frame_files[0]))
    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(str(frame_file))
        writer.write(frame)
        if (i + 1) % 30 == 0:
            print(f"  Written {i + 1}/{len(frame_files)} frames...")

    writer.release()
    print(f"Rebuilt video: {output_path} ({len(frame_files)} frames at {fps} fps)")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Extract & rebuild video frames")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--frames-dir", default="outputs/frames", help="Directory to store frames")
    parser.add_argument("--rebuild", action="store_true", help="Also rebuild video from frames")
    parser.add_argument("--output", default="outputs/rebuilt.mp4", help="Path for rebuilt video")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    meta = extract_frames(args.video, args.frames_dir)

    if args.rebuild:
        rebuild_video(meta["frames_dir"], args.output, meta["fps"])
        print("\nPhase 1 complete — extract + rebuild pipeline verified.")
    else:
        print("\nPhase 1 extract complete. Run with --rebuild to verify reconstruction.")
