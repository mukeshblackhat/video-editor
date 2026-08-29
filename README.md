# Video Editor — Change the Background of Any Video

This tool takes a video of a person, cuts them out, and puts them on a new
background image. Think of it like a green screen, but you don't need a green
screen.

It uses two AI models:

- **SAM 2** (from Meta) — finds the person in the video.
- **MatAnyone** — draws a soft, detailed outline around them, so hair and edges
  look natural instead of cut out with scissors.

### The Difference in One Line

**SAM 2 says a pixel is either 0 or 1 — keep it or drop it. MatAnyone says a
pixel can be anything between 0 and 1 — how visible it is.**

That in-between is everything. A strand of hair is not fully there and not fully
gone; it is maybe 30% visible with the background showing through. SAM 2 has to
pick one or the other, so hair turns into a jagged, chopped edge. MatAnyone
stores the 30%, so hair, soft edges, and motion blur come out looking real.

## What You Need

- Python 3.10 or newer
- `ffmpeg` installed on your computer (on Mac: `brew install ffmpeg`)
- A GPU helps a lot (Apple MPS or NVIDIA CUDA). It works on CPU too, but slowly.
- About 1 GB of free space for the model files

## Setup

Run this once:

```bash
git clone git@github.com:mukeshblackhat/video-editor.git
cd video-editor
bash setup.sh
source venv/bin/activate
```

This makes a virtual environment, installs everything, and downloads the SAM 2
model file (~2.4 GB). The MatAnyone model (~141 MB) downloads by itself the
first time you run the pipeline.

Everything lives on `main` — MatAnyone matting is on by default, so there is no
branch to switch to.

## How to Use It

Put your video and your background image in the `inputs/` folder, then run:

```bash
python run_pipeline.py --video inputs/video.mp4 --background inputs/bg.jpg
```

The first frame of your video will open in a window. Click on the person you
want to keep, then close the window. The tool does the rest.

If you already know where the person is, you can skip the clicking and give the
coordinates directly:

```bash
python run_pipeline.py --video inputs/video.mp4 --background inputs/bg.jpg --points "320,240"
```

You can give more than one point if one click isn't enough:

```bash
python run_pipeline.py --video inputs/video.mp4 --background inputs/bg.jpg --points "320,240,400,300"
```

When it finishes, your new video is at `outputs/final.mp4`, with the original
audio kept.

## How It Works (5 Steps)

1. **Extract** (`phase1_extract.py`) — Splits the video into single picture
   frames.
2. **Cut out the person** (`phase2b_matting.py`) — SAM 2 finds the person in the
   first frame, then MatAnyone follows them through the rest of the video and
   makes a soft mask for each frame.
3. **Composite** (`phase3_composite.py`) — Places the person on the new
   background, cleans up the edges, and matches the colors and lighting so it
   looks like one real photo.
4. **Render** (`phase4_render.py`) — Joins the frames back into an MP4 and adds
   the original audio.

`run_pipeline.py` runs all of these in order for you.

### The Extra Polish Modules

These run automatically as part of step 3:

- **`edge_refine.py`** — Makes the outline smooth and removes color bleeding
  from the old background.
- **`color_harmonize.py`** — Matches the person's brightness and color to the
  new background, so they don't look pasted on.
- **`bg_motion.py`** — Adds slow, gentle movement to the background (zoom, pan,
  parallax) so it doesn't look like a still photo.

## Useful Options

| Option | What it does |
| --- | --- |
| `--points "x,y"` | Skip the click window, give coordinates instead |
| `--matting-mode sam2` | Use the older SAM 2 masks instead of MatAnyone |
| `--no-edge-refine` | Turn off edge smoothing |
| `--no-color-harmonize` | Turn off color and lighting matching |
| `--no-decontaminate` | Turn off old-background color removal |
| `--no-bg-motion` | Keep the background completely still |
| `--harmonize-strength 0.6` | How strongly to match colors (0 to 1) |
| `--blur-radius 5` | How soft the edges are |

Every step can also be run on its own — look at the bottom of each file.

## Where Files Go

```
inputs/                  your video and background image
outputs/frames/          the original frames, one per picture
outputs/masks/           the cut-out shape for each frame
outputs/composited/      frames with the new background
outputs/final.mp4        your finished video
checkpoints/             the AI model files
```

## If Something Goes Wrong

- **No audio in the output** — `ffmpeg` isn't installed. Install it and run
  again.
- **Out of memory** — Use `--max-matting-size 720` to process at a smaller size.
- **The wrong thing got cut out** — Add more click points with `--points`, or
  click more carefully on the person in the window.
- **It's very slow** — You're probably running on CPU. That's expected.
