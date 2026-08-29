"""
Full Pipeline: Video Background Replacement using SAM 2 + MatAnyone

Usage:
    # Default mode (MatAnyone soft alpha — best quality):
    python run_pipeline.py --video inputs/video.mp4 --background inputs/bg.jpg --points "320,240"

    # Legacy mode (SAM 2 binary masks only):
    python run_pipeline.py --video inputs/video.mp4 --background inputs/bg.jpg --points "320,240" --matting-mode sam2

    # Fast mode (disable all enhancements):
    python run_pipeline.py --video inputs/video.mp4 --background inputs/bg.jpg --points "320,240" \\
        --matting-mode sam2 --no-edge-refine --no-color-harmonize --no-bg-motion
"""

import os
import argparse
from pathlib import Path

from phase1_extract import extract_frames
from phase3_composite import composite_all_frames
from phase4_render import render_video, add_audio


def run_full_pipeline(video_path: str, background_path: str, checkpoint: str,
                      model_cfg: str, points: str = None, blur_radius: int = 5,
                      edge_refine: bool = True, edge_width: int = 15,
                      color_harmonize: bool = True, harmonize_strength: float = 0.6,
                      decontaminate: bool = True, decontaminate_strength: float = 0.7,
                      bg_motion: bool = True, matting_mode: str = "matanyone",
                      n_warmup: int = 10, r_erode: int = 10, r_dilate: int = 10,
                      max_matting_size: int = -1):
    """Run the complete background replacement pipeline.

    Args:
        video_path: Path to input video.
        background_path: Path to new background image.
        checkpoint: SAM 2 checkpoint path.
        model_cfg: SAM 2 model config.
        points: Foreground click coordinates as 'x1,y1,x2,y2,...'.
        blur_radius: Edge feathering radius.
        edge_refine: Enable edge-aware mask refinement.
        edge_width: Edge transition band width.
        color_harmonize: Enable color/lighting harmonization.
        harmonize_strength: Harmonization strength [0-1].
        decontaminate: Enable edge color decontamination.
        decontaminate_strength: Decontamination strength [0-1].
        bg_motion: Enable background movement.
        matting_mode: 'matanyone' (default, soft alpha) or 'sam2' (binary masks).
        n_warmup: MatAnyone warmup iterations.
        r_erode: MatAnyone mask erosion radius.
        r_dilate: MatAnyone mask dilation radius.
        max_matting_size: Max frame dim for MatAnyone (-1 = no limit).
    """
    base_dir = Path(__file__).parent
    os.chdir(base_dir)

    frames_dir = "outputs/frames"
    masks_dir = "outputs/masks"
    composited_dir = "outputs/composited"
    output_video = "outputs/final.mp4"

    if not points:
        raise ValueError(
            "Please provide --points 'x,y' for foreground click coordinates.")

    # Parse points
    coords = list(map(int, points.split(",")))
    point_list = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    prompt = {"points": point_list, "labels": [1] * len(point_list)}

    # Phase 1: Extract frames
    print("=" * 60)
    print("PHASE 1: Extracting frames")
    print("=" * 60)
    meta = extract_frames(video_path, frames_dir)

    # Phase 2: Segmentation / Matting
    print("\n" + "=" * 60)
    if matting_mode == "matanyone":
        print("PHASE 2: SAM 2 first-frame + MatAnyone video matting (soft alpha)")
        print("=" * 60)
        from phase2b_matting import run_full_matting_pipeline
        run_full_matting_pipeline(
            frames_dir, prompt, checkpoint, model_cfg, masks_dir,
            n_warmup=n_warmup, r_erode=r_erode, r_dilate=r_dilate,
            max_long_side=1024, max_size=max_matting_size,
        )
    else:
        print("PHASE 2: SAM 2 segmentation (binary masks)")
        print("=" * 60)
        from phase2_segment import run_segmentation
        run_segmentation(frames_dir, prompt, checkpoint, model_cfg, masks_dir)

    # Phase 3: Composite
    print("\n" + "=" * 60)
    print("PHASE 3: Compositing frames")
    print("=" * 60)
    composite_all_frames(
        frames_dir, masks_dir, background_path, composited_dir,
        blur_radius=blur_radius,
        edge_refine=edge_refine,
        edge_width=edge_width,
        color_harmonize=color_harmonize,
        harmonize_strength=harmonize_strength,
        decontaminate=decontaminate,
        decontaminate_strength=decontaminate_strength,
        bg_motion=bg_motion,
    )

    # Phase 4: Render
    print("\n" + "=" * 60)
    print("PHASE 4: Rendering final video")
    print("=" * 60)
    silent_path = "outputs/final_no_audio.mp4"
    render_video(composited_dir, silent_path, meta["fps"])
    add_audio(silent_path, video_path, output_video)

    print("\n" + "=" * 60)
    print(f"DONE! Output: {Path(output_video).resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video Background Replacement using SAM 2 + MatAnyone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Best quality (MatAnyone soft alpha + all enhancements):
  python run_pipeline.py --video inputs/video.mp4 --background inputs/bg.jpg --points "320,240"

  # Legacy SAM 2 binary masks:
  python run_pipeline.py --video inputs/video.mp4 --background inputs/bg.jpg --points "320,240" \\
      --matting-mode sam2

  # Fast mode (minimal processing):
  python run_pipeline.py --video inputs/video.mp4 --background inputs/bg.jpg --points "320,240" \\
      --matting-mode sam2 --no-edge-refine --no-color-harmonize --no-bg-motion
        """,
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--background", required=True, help="New background image path")
    parser.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_large.pt",
                        help="SAM 2 checkpoint path")
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml",
                        help="SAM 2 model config")
    parser.add_argument("--points", help="Foreground click points as 'x1,y1,x2,y2,...'")
    parser.add_argument("--matting-mode", choices=["matanyone", "sam2"],
                        default="matanyone",
                        help="Matting mode: 'matanyone' (soft alpha, default) or 'sam2' (binary)")
    parser.add_argument("--blur-radius", type=int, default=5, help="Mask edge feathering")
    parser.add_argument("--edge-width", type=int, default=15,
                        help="Edge transition band width in pixels")
    parser.add_argument("--no-edge-refine", action="store_true",
                        help="Disable edge-aware mask refinement")
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
    # MatAnyone-specific args
    parser.add_argument("--n-warmup", type=int, default=10,
                        help="MatAnyone warmup iterations (default 10)")
    parser.add_argument("--r-erode", type=int, default=10,
                        help="MatAnyone mask erosion radius (default 10)")
    parser.add_argument("--r-dilate", type=int, default=10,
                        help="MatAnyone mask dilation radius (default 10)")
    parser.add_argument("--max-matting-size", type=int, default=-1,
                        help="Max frame dim for MatAnyone processing (-1 = no limit)")
    args = parser.parse_args()

    run_full_pipeline(
        args.video, args.background, args.checkpoint, args.model_cfg,
        args.points, args.blur_radius,
        edge_refine=not args.no_edge_refine,
        edge_width=args.edge_width,
        color_harmonize=not args.no_color_harmonize,
        harmonize_strength=args.harmonize_strength,
        decontaminate=not args.no_decontaminate,
        decontaminate_strength=args.decontaminate_strength,
        bg_motion=not args.no_bg_motion,
        matting_mode=args.matting_mode,
        n_warmup=args.n_warmup,
        r_erode=args.r_erode,
        r_dilate=args.r_dilate,
        max_matting_size=args.max_matting_size,
    )
