"""
Phase 4: Render — Assemble composited frames into final video
- Read composited frames
- Write to MP4 using ffmpeg (H.264) for maximum compatibility
- Fallback to OpenCV VideoWriter if ffmpeg is unavailable
- Copy audio from original video
"""

import cv2
import subprocess
import os
import argparse
import shutil
from pathlib import Path


def render_video(composited_dir: str, output_path: str, fps: float):
    """Render composited frames into video.

    Attempts ffmpeg with H.264 first for maximum browser/device compatibility.
    Falls back to OpenCV VideoWriter (mp4v) if ffmpeg is unavailable.
    """
    composited_dir = Path(composited_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(composited_dir.glob("comp_*.png"))
    if not frame_files:
        raise ValueError(f"No composited frames in {composited_dir}")

    first = cv2.imread(str(frame_files[0]))
    h, w = first.shape[:2]

    # Try ffmpeg H.264 first
    if _render_with_ffmpeg(composited_dir, output_path, fps, w, h):
        print(f"Video rendered (H.264): {output_path} ({len(frame_files)} frames, {fps} fps)")
        return str(output_path)

    # Fallback to OpenCV
    print("ffmpeg not available — falling back to OpenCV VideoWriter (mp4v codec)")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for i, f in enumerate(frame_files):
        frame = cv2.imread(str(f))
        if frame is None:
            raise ValueError(f"Cannot read composited frame: {f}")
        writer.write(frame)
        if (i + 1) % 30 == 0:
            print(f"  Rendered {i + 1}/{len(frame_files)} frames...")

    writer.release()
    print(f"Video rendered (mp4v): {output_path} ({len(frame_files)} frames, {fps} fps)")
    return str(output_path)


def _render_with_ffmpeg(composited_dir: Path, output_path: Path,
                        fps: float, width: int, height: int) -> bool:
    """Render using ffmpeg with H.264 codec. Returns True on success."""
    # ffmpeg reads PNG sequence with glob pattern
    input_pattern = str(composited_dir / "comp_%06d.png")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg H.264 encoding failed: {e.stderr[:200]}")
        return False


def add_audio(video_no_audio: str, original_video: str, final_output: str):
    """Copy audio from original video to the new video using ffmpeg."""
    final_output = Path(final_output)
    final_output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", video_no_audio,
        "-i", original_video,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        str(final_output),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Audio merged: {final_output}")
    except FileNotFoundError:
        print("ffmpeg not found — output video has no audio.")
        print(f"Install ffmpeg and run: ffmpeg -i {video_no_audio} -i {original_video} "
              f"-c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest {final_output}")
        shutil.copy2(video_no_audio, str(final_output))
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e.stderr}")
        shutil.copy2(video_no_audio, str(final_output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: Render final video")
    parser.add_argument("--composited-dir", default="outputs/composited", help="Composited frames")
    parser.add_argument("--output", default="outputs/final.mp4", help="Output video path")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS")
    parser.add_argument("--original-video", help="Original video (to copy audio from)")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    silent_path = args.output if not args.original_video else "outputs/final_no_audio.mp4"
    render_video(args.composited_dir, silent_path, args.fps)

    if args.original_video:
        add_audio(silent_path, args.original_video, args.output)

    print(f"\nDone! Final video: {args.output}")
