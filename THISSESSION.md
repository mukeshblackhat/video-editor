# Session Log — Pipeline Performance Work

Tracking what we decided, why, and where we're heading.

## Where we started

User's report: the pipeline "takes too much time to run." Asked for a
read-through of the pipeline first, then a plan to cut total runtime.

## What the pipeline does (established by reading the code)

4 phases orchestrated by `run_pipeline.py`:

1. **phase1_extract.py** — OpenCV decodes every frame, writes each as PNG.
2. **phase2b_matting.py** (default) — SAM 2 produces a first-frame mask from
   the user's click; MatAnyone propagates it across all frames as a soft alpha
   matte (continuous 0-255, not binary). Legacy binary path is
   `phase2_segment.py` via `--matting-mode sam2`.
3. **phase3_composite.py** — per frame: edge refine, LAB decontamination,
   background motion, color harmonization, alpha blend, write PNG.
4. **phase4_render.py** — ffmpeg H.264 `-crf 18 -preset medium`, then merge
   original audio.

Design quality is good — soft alpha handled correctly, background resize
cached, proper rounding not truncation, every enhancement individually
togglable. The architecture is not the problem.

## Root cause of the slowness

The input is **2160x3840 (4K portrait), 911 frames, 24fps** (~38s of video).
The pipeline runs everything at full 4K through a PNG-on-disk architecture.

Measured evidence in `outputs/`:

| Path | Size |
|---|---|
| `outputs/frames` | 5.4 GB (911 PNGs, ~5.6 MB each) |
| `outputs/composited*` (4 runs) | 2.0 + 4.1 + 3.8 + 7.9 GB |
| `outputs/full_run` | 21 GB |
| **Total** | **~44 GB** |

Four costs, in order of impact:

1. **PNG round-trips** — every frame is PNG-encoded in p1, decoded in p3,
   re-encoded in p3, decoded in p4. Four 8-megapixel codec passes per frame,
   ~3,600 total, single-threaded.
2. **Phase 3 runs at 4K** — guided filter, 2x distanceTransform, several LAB
   cvtColor round-trips, LANCZOS4 resize of a 16 MB background, filter2D
   motion blur, all on 8.3M pixels per frame in Python.
   Note: when MatAnyone is used, phase 3 detects soft alpha and skips
   `refine_mask_edges` entirely (`phase3_composite.py:141`) — so edge-refine
   cost only applies on the `sam2` path.
3. **MatAnyone defaults to no downscaling** — `--max-matting-size` defaults to
   `-1`, so the network does full 4K inference on all 911 frames on MPS.
   `run_pipeline.py` hardcodes `max_long_side=1024` for the SAM 2 first frame
   but passes `-1` straight through to MatAnyone.
4. **Everything is sequential** — phase 3 is a plain `for` loop, no
   multiprocessing, despite being embarrassingly parallel.

## Environment (measured, not assumed)

- 12 cores (8 performance + 4 efficiency), 24 GB RAM
- ffmpeg 8.1 with `h264_videotoolbox` / `hevc_videotoolbox` hardware encoders
- Source audio is AAC

## Benchmarks (measured this session, per frame)

Codec and resize costs:

| Operation | Time |
|---|---|
| PNG encode 4K | 93.5 ms |
| JPEG q95 encode 4K | 11.3 ms |
| JPEG q95 encode 1080p | 3.1 ms |
| PNG decode 4K | 61.8 ms |
| LANCZOS4 resize 4K | 0.4 ms |
| cvtColor BGR2LAB 4K | 14.9 ms |
| cvtColor BGR2LAB 1080p | 3.7 ms |

Phase 3 enhancement stack:

| Operation | Time |
|---|---|
| `apply_background_motion` | **475.9 ms** |
| `harmonize_colors` | **386.6 ms** |
| `np.unique(mask)` soft-alpha check | 35.7 ms |
| `decontaminate_edges` | 4.4 ms |

Two findings worth calling out:

- **`apply_background_motion` is the single largest per-frame cost.** It
  transforms the background at its native 5432x3072 and only *then* downscales
  to frame size. The work is being done at ~3x the pixels it needs.
- **`np.unique` costs 35.7 ms per frame** purely to answer a yes/no question
  ("is this soft alpha?"). The answer is identical for every frame in a run.

## Decisions (user's answers)

| Question | Decision | Consequence |
|---|---|---|
| Output resolution | **1080p (1080x1920) is fine** | Everything after decode runs at 1/4 the pixels. Largest single win. |
| Intermediate frames | **Keep, but as fast JPEG q95** | Still inspectable on disk; ~10x faster encode, ~15x smaller than PNG. |
| Enhancement stack | **Keep all, add a `--fast` preset** | Default visual output preserved; one flag for quick preview runs. |
| CLI compatibility | **Keep backward compatible** | All existing flags keep working with the same meaning. New options added alongside. |

## Where we're heading

A phased optimization plan, to be presented for approval before any code is
written (per CLAUDE.md: zero assumptions, phased plan, test-first).

Optimization levers identified, ordered by expected impact:

1. Process at 1080p instead of 4K after decode.
2. Replace PNG intermediates with JPEG q95.
3. Do background motion at target resolution, not native background resolution.
4. Hoist the soft-alpha detection out of the per-frame loop.
5. Parallelize phase 3 across the 12 cores.
6. Run MatAnyone at reduced resolution, upscale the alpha matte back (alpha
   upscales far more gracefully than color detail).
7. Consider `h264_videotoolbox` for hardware-accelerated encoding.

## Constraints carried from CLAUDE.md

- Test-first: write the failing test before the implementation, every time.
- Verify after every change; run the full relevant suite before declaring a
  phase complete.
- Backward compatibility is required unless the user explicitly agrees
  otherwise — and here they explicitly asked for it.
- Minimal blast radius: no refactoring of unrelated code.

## Branch

Work happens on **`fast`**, branched from `main` at `6aa5329`.

## Estimated current runtime (derived from the measured per-frame costs)

For 911 frames:

| Phase | Per frame | Total |
|---|---|---|
| Phase 1 (decode + PNG encode) | 123.5 ms | ~1.9 min |
| Phase 3 (full enhancement stack) | 1144.7 ms | **~17.4 min** |

Phase 3 breakdown by share of its own time:

| Component | Share |
|---|---|
| `apply_background_motion` | 41.6% |
| `harmonize_colors` | 33.8% |
| PNG decode + encode | 19.0% |
| `np.unique` soft-alpha check | 3.1% |
| decontaminate + blend | ~2.5% |

**Phases 1 and 3 alone come to ~19.3 minutes**, before counting MatAnyone's
full-4K inference across 911 frames or the final `-preset medium -crf 18`
encode. Phase 3 is the dominant cost and the primary target.

Additional finding while reading the hot functions: `harmonize_colors` runs
four sequential sub-passes (histogram, white balance, exposure, ambient cast),
each doing its own BGR->LAB->BGR round-trip — roughly 8 color conversions per
frame where 2 would suffice. Consolidating to a single LAB round-trip is a
clean win that changes no output.

## Correctness issue found: background aspect ratio is ignored

User raised this: when replacing the background, the video's aspect ratio and
the background's aspect ratio must match. Investigated — **the current code is
unconditionally wrong here.**

Both background sizing call sites — `phase3_composite.py:110` (the cached
resize) and `phase3_composite.py:159` (the per-frame post-motion resize) — do:

```python
cv2.resize(background, (w, h), interpolation=cv2.INTER_LANCZOS4)
```

This is a straight stretch to frame dimensions. It never consults the
background's aspect ratio, so a mismatched background is anisotropically
distorted (circles -> ellipses, verticals lean).

### Why it hasn't shown up yet

Video AR is 0.5625 (2160x3840, 9:16 portrait). All three current backgrounds
are already near-9:16:

| Background | Size | AR | Distortion today |
|---|---|---|---|
| `bg2.png` | 3072x5432 | 0.5655 | 0.5% |
| `custom_bg.png` | 2368x4192 | 0.5649 | 0.4% |
| `white_bg.png` | 2160x3840 | 0.5625 | 0.0% |

Sub-1% stretch is invisible. **The bug is latent, not absent.** A 16:9
landscape background (AR 1.778) would be squeezed to 0.5625 — a 68% horizontal
compression — with no warning from the code.

Second instance: `ken_burns_transform` crops at the *background's* aspect
ratio, and `phase3_composite.py:159` then stretches that result to frame
dims — so distortion is applied *after* the motion effect, every frame.

### Decided fix

A `fit_background_to_frame()` helper in `bg_motion.py`, used by both call
sites, exposed as `--bg-fit`:

- **`cover`** (new default) — scale to fill, center-crop the overflow.
  Preserves geometry, no letterboxing. For the current backgrounds it discards
  only 0.4-0.5%, so output stays essentially identical to today.
- **`stretch`** — current behavior, retained for backward compatibility.
- **`contain`** — fit entirely, pad the remainder.

Two details to get right:
1. The cover-crop must be computed **before** Ken Burns, not after, so motion
   operates in the correct aspect space.
2. Apply the crop once and cache it; do not recompute per frame.

Also add a startup warning when the AR mismatch exceeds ~2%, so a badly matched
background is caught immediately instead of after a full-length run.

Cost: negligible — a crop is a view, not a copy. Folded into the plan as its
own phase.

## Switched input: Diksha Sethi submission + dark purple background

User redirected to a video from `~/Downloads/cred_video_submissions 2/`
(9 personal video submissions). Confirmed intent: use their video as the **new
pipeline input** and replace their background — not composite anyone into
someone else's footage.

Selected **Diksha Sethi**: 1080x1920, AR 0.5625, 24fps, 309 frames, 12.9s.
Copied to `inputs/diksha/diksha.mp4`.

### Background: generated three dark plates, user chose purple

All at **1080x1920, AR 0.5625** — exact match to the video, so no crop and no
stretch. Soft lit pool behind the head, rim glow upper-right, darkening toward
the bottom, plus grain (dark backgrounds band badly under H.264 without it).

| File | Mean L* |
|---|---|
| `inputs/dark_green_bg.png` | 39.6 |
| `inputs/dark_purple_bg.png` | 33.1 <- **chosen** |
| `inputs/dark_orange_bg.png` | 54.3 |

### Click points — a mistake worth remembering

First attempt used `--points "560,1020"`, which landed on her **glasses**. SAM 2
segmented just the eyewear: 2.2% frame coverage, bbox 443-867 x 975-1116.
Caught it because coverage was implausibly low for a talking-head shot.

Used an OpenCV Haar face detector to locate the true face center (645,1095),
then re-ran with three torso points: `--points "645,1250,645,1650,430,1500"`.
Result: **41.7% coverage, 1.64% soft edge pixels** — a correct full-person
matte with real hair detail.

**Lesson: for a subject wearing glasses, click the torso, not the face.**

### Baseline run

Running the **unmodified** pipeline to establish both baselines at once:
- the **quality** bar the optimization work must not regress
- the **timing** bar it must beat

Command:
```
python run_pipeline.py --video inputs/diksha/diksha.mp4 \
  --background inputs/dark_purple_bg.png \
  --points "645,1250,645,1650,430,1500"
```

Prior 4K outputs preserved as `outputs/*_4k_orig` before this run, since the
pipeline hardcodes output paths.

Note: the aspect-ratio bug does **not** trigger here — the background matches
the video exactly. The fix is still pending and still needed.

### Phase 3 profile at 1080x1920 (differs from the 4K profile)

Re-profiled on Diksha's frames, since the cost mix changes with resolution
and with how much soft edge the matte contains:

| Operation | 4K (input.mp4) | 1080p (diksha) |
|---|---|---|
| `harmonize_colors` | 386.6 ms | **168.4 ms** <- now dominant |
| `apply_background_motion` | 475.9 ms | 89.9 ms |
| `decontaminate_edges` | 4.4 ms | **79.4 ms** <- 18x jump |
| PNG encode | 93.5 ms | 32.6 ms |
| PNG decode x2 | 123.6 ms | 26.3 ms |
| `np.unique` | 35.7 ms | 10.2 ms |

Two shifts worth noting:

1. **`apply_background_motion` is no longer the top cost.** At 4K it dominated
   because the background was 5432x3072 (bigger than the frame). Here the
   background is exactly 1080x1920, so there is no oversized-source penalty.
   This confirms the diagnosis: its 4K cost was about operating on a background
   larger than the target, not about the motion math.

2. **`decontaminate_edges` jumped 18x** (4.4 -> 79.4 ms) despite the *smaller*
   frame. It scales with the number of soft edge pixels, not frame area — this
   matte has loose hair against a plain wall, so far more pixels land in the
   `0.1 < alpha < 0.9` band. Optimizing it matters more than the 4K profile
   suggested.

**MatAnyone is the real bottleneck for this clip.** Running at full 1080x1920
with `max_size=-1` on MPS, it is producing roughly 10s/frame, so phase 2 alone
is ~50 min for 309 frames — far more than all of phase 3. This raises the
priority of running matting at reduced resolution and upscaling the alpha.

## BASELINE COMPLETE — 16m 19s (979s), exit 0

309 frames through all four phases. Output verified: H.264 1080x1920, 24fps,
AAC audio, 12.84s, 5.7MB.

### Measured per-phase timing (from file mtimes, not estimates)

| Phase | Time | Per frame | Share |
|---|---|---|---|
| 1 extract | 10s | 0.03s | 1% |
| **2 MatAnyone matting** | **846s** | **2.74s** | **86%** |
| 3 composite | 110s | 0.36s | 11% |
| 4 render | 3s | — | 0.3% |

Two corrections to earlier projections:
- Phase 2 is **86%**, even more dominant than the 83% projected.
- Phase 4 took **3 seconds**, not the ~1 min estimated. **Hardware-encode
  (VideoToolbox) is dropped from the plan — there is nothing there to win.**

### Matte quality: excellent and stable

| Frame | Coverage | Soft px |
|---|---|---|
| 0 | 41.7% | 1.56% |
| 75 | 42.1% | 1.88% |
| 150 | 42.1% | 1.88% |
| 225 | 42.0% | 1.95% |
| 308 | 41.8% | 1.94% |

No drift, no collapse across 309 frames. MatAnyone's temporal memory works.

## DEFECT FOUND: harmonize_colors recolors the whole subject

The delivered baseline has a heavy purple wash over the subject's skin and
olive shirt. Diagnosed by measuring pixels where **alpha = 1.0** (solid
interior — compositing must leave these untouched):

| Pipeline | Mean color shift on solid interior |
|---|---|
| Plain alpha blend, no enhancements | **0.00** (mathematically correct) |
| Full baseline pipeline | **43.3** (peak delta 181) |
| `harmonize_colors` alone | ~50 in G and R |

`harmonize_colors` is the sole cause. It matches the foreground's color
distribution to the background's — reasonable against a *photographic* plate,
but against a **saturated dark purple** background it drags the person toward
purple. The dark, saturated target makes the histogram/exposure passes
overcorrect badly.

### Strength does not scale the defect away

| Strength | Subject shift |
|---|---|
| 0.6 (default) | 43.26 |
| 0.3 | 35.85 |
| 0.15 | 34.16 |
| 0.0 (off) | 0.37 |

Shift barely moves from 0.6 -> 0.15, then collapses at 0. The four sub-passes
do not scale down linearly with the master strength — so turning the dial down
is **not** an effective fix. It is close to on/off.

### Why this matters for the goal

The user's goal is "improve speed without losing the quality we got last time."
The quality they approved was the **plain-blend preview** (natural skin, olive
shirt). The baseline is **worse** than that preview. Optimizing against this
baseline would mean preserving a defect.

**Recommendation: `--no-color-harmonize` for saturated backgrounds.** It both
looks correct and removes 168ms/frame from phase 3. Pending user decision.
