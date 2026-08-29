# Pipeline Results Log

Every change gets a row. Timings are wall-clock on the same machine
(M-series, 12 cores, 24 GB); quality is measured, not eyeballed.

**Fixed test clip:** Diksha Sethi, 1080x1920, 24 fps, 309 frames, 12.9 s
**Background:** `inputs/dark_purple_bg.png` (1080x1920, AR 0.5625, exact match)
**Points:** `645,1250,645,1650,430,1500`

## Metrics and why they are the ones we track

| Metric | What it catches | Good value |
|---|---|---|
| **Total time** | The headline goal | lower |
| **Per-phase time** | Which phase a change actually moved | — |
| **Subject shift** | Mean abs BGR change where alpha=1.0. Compositing must NOT alter solid interior. | **~0** |
| **Edge light retained** | % of the original bright wall still in the hair edge. This is the white-halo defect. | **low** |
| **Soft px %** | Fraction of pixels with 0<alpha<255 — the hair detail MatAnyone found. Collapse = lost hair. | **~1.9%** |
| **Coverage %** | Fraction of frame matted as subject. Sudden change = matte broke. | **~42%** |

## Run history

### R0 — Baseline (unmodified pipeline)

Commit: `6aa5329` (main) | `--matting-mode matanyone`, all enhancements on

| Phase | Time | Share |
|---|---|---|
| 1 extract | 10 s | 1% |
| 2 MatAnyone matting | **846 s** | **86%** |
| 3 composite | 110 s | 11% |
| 4 render | 3 s | 0.3% |
| **TOTAL** | **979 s (16m 19s)** | |

| Quality | Value | Verdict |
|---|---|---|
| Subject shift | **42.04** | **FAIL** — whole subject washed purple |
| Coverage | 41.7–42.1% | good, stable |
| Soft px | 1.56–1.95% | good |
| Edge light retained | 46% | **FAIL** — white halo on hair |

Output: `outputs/final.mp4` (5.7 MB)

**Defects found:** (1) `harmonize_colors` recolors the entire subject against a
saturated background; (2) original bright-wall light is retained in the hair
edge, reading as a white halo against dark purple.

---

### R1 — Harmonization off

Change: `--no-color-harmonize` (flag only, no code change)

| Phase | Time | vs R0 |
|---|---|---|
| 3 composite | **67 s** | **−43 s (−39%)** |
| TOTAL (est.) | ~936 s | −43 s |

| Quality | R0 | R1 |
|---|---|---|
| Subject shift | 42.04 | **0.37** ✅ |
| Edge light retained | 46% | 46% (unchanged — separate defect) |

Output: `outputs/final_noharm.mp4` (11 MB)

**Result:** Better quality *and* faster. Subject is now mathematically
untouched where alpha=1. The white halo remains — it is a spill problem, not a
harmonization one.

Larger file than R0 because the purple wash had flattened detail; more
retained detail compresses less.

---

### R2 — Spill fix + matting speedup (in progress)

Two changes together:
1. **Spill suppression** in `edge_refine.py` — remove the original
   background's light from semi-transparent edge pixels.
2. **`--max-matting-size`** — run MatAnyone at reduced resolution and upscale
   the alpha, targeting the 846 s that is 86% of the run.

Results to follow.

#### R2a — Matting resolution A/B (40-frame slice)

Measured before committing, because this is the one change that can cost real
quality. Reference is the full-resolution matte.

| Setting | 40-frame time | Projected 309 | Coverage | Soft px | Edge IoU | **hairMAE** |
|---|---|---|---|---|---|---|
| full (ref) | 846 s (full run) | 14.1 min | 42.0% | 1.65% | 100% | 0.0 |
| **768** | **21 s** | **2.7 min** | 41.8% | 2.62% | **99.4%** | **25.9** |
| 512 | 11 s | 1.4 min | 41.8% | 2.92% | 99.1% | 34.3 |

`hairMAE` = mean alpha error restricted to the reference's soft band, i.e.
only where hair detail actually lives. The silhouette agrees to >99% at both
sizes — the body outline is never the problem. The question is only the hair.

**Chose 768.** Visual inspection at 1.4x on the hair region is decisive:
full resolution shows crisp individual strands; 768 keeps strand structure
with slight softening; 512 is visibly blurrier, the soft band widens and
strands merge together. 512 is 2x faster still, but it loses exactly the
detail this pipeline exists to preserve.

This is why the A/B was worth running: the aggregate metrics (coverage, IoU)
look nearly identical at both sizes and would have justified 512. Only the
hair-band metric and the zoomed visual show the real cost.

### R2 — Spill fix + matting at 768 (clean run from zero)

Commits `e0e078b` + `a8d8baf` | `--max-matting-size 768 --no-color-harmonize`

**TOTAL: 234 s (3m 54s) vs baseline 979 s (16m 19s) — 4.2x faster, 12.4 min saved**

| Phase | R2 | R0 baseline | Speedup |
|---|---|---|---|
| 1 extract | 10 s | 10 s | 1.0x |
| **2 matting** | **132 s** | **846 s** | **6.4x** |
| 3 composite | 80 s | 110 s | 1.4x |
| 4 render | ~3 s | 3 s | 1.0x |

Phase 2 dropped from 86% of the run to 56%. MatAnyone ran at 768x1365
(downscaled from 1080x1920), alpha upscaled back to full resolution.

Phase 3 got faster *despite adding* spill suppression, because the background
is now fitted once instead of a per-frame INTER_LANCZOS4 resize.

| Quality | R0 | R2 | Verdict |
|---|---|---|---|
| Subject shift (alpha=1) | 41.65 | **0.37** | ✅ subject untouched |
| Edge/subject brightness ratio | 1.013 | **0.680** | ✅ halo gone |
| Coverage | 42.0% | 41.6–41.9% | ✅ stable |
| Soft px | 1.65% | 2.64–3.13% | ✅ more edge detail |

Output: `outputs/final.mp4` (11 MB)

#### A metric that misled, and the correction

The first comparison used "edge light retained" as an absolute percentage and
showed the halo getting *worse* (10.1% -> 16.4%). That reading was wrong.

`harmonize_colors` in R0 had darkened the **entire** subject — core brightness
58.1 against a true 95.6 — so the edge looked dark only because everything was
dark. The fair test is the edge measured **relative to the subject it belongs
to**:

| | Edge / subject brightness |
|---|---|
| R0 baseline | **1.013** — edge is *brighter* than the subject = halo |
| R2 | **0.680** — edge is darker than the subject = correct |

Lesson: an absolute brightness metric is meaningless when a change also shifts
the overall exposure. Always normalize against something the change did not move.

### R3 — Second subject validation: Manvi Mehta

Same optimized settings, same purple background, a different and harder clip.
Purpose: confirm the changes generalize rather than fitting one video.

`--max-matting-size 768 --no-color-harmonize`, points `501,946,501,1051,501,1400`

| | Diksha | Manvi |
|---|---|---|
| Frames | 309 (12.9 s) | **359 (15.0 s)** |
| Original background | cream wall | **beige curtain** |
| Shot | close-up | **full body standing** |
| Hair | straight | **long curly** |
| Clothing | olive | **lavender + white** |

| Phase | Time |
|---|---|
| 1 extract | 15 s |
| 2 matting | 122 s |
| 3 composite | 103 s |
| **TOTAL** | **254 s (4m 14s)** |

**0.708 s/frame vs Diksha's 0.757 s/frame** — slightly faster per frame on a
harder clip. The optimization is not clip-specific.

| Quality | Value |
|---|---|
| Coverage | 25.1–28.0% (lower than Diksha's 42% — she stands further from camera) |
| Soft px | 1.59–1.68% |
| Subject shift | **0.40–0.42** ✅ |
| Edge/subject ratio | **0.47–0.50** ✅ (better than Diksha's 0.68) |

The edge ratio is *better* here despite a brighter original background, because
the beige curtain is more uniform than Diksha's wall-plus-curtain, so the single
background colour estimate in `suppress_edge_spill` fits it more closely.

Outputs kept as `outputs/final_diksha.mp4` and `outputs/final_manvi.mp4`, with
per-subject `frames_*`, `masks_*`, `composited_*` directories.

---

## R4 — MatAnyone 2 evaluation (branch `matanyone2`)

MatAnyone 2 (CVPR 2026 Highlight) released Dec 2025. Paper claims −26% MAD and
−24.5% gradient error vs v1 on real-world benchmarks. Evaluated on our clip.

### Getting it installed — two upstream problems

1. **The documented install is broken.** `pip install matanyone2@git+...` fails:
   `ValueError: A second file is being added to the wheel archive at the same
   path: matanyone2/config/__init__.py`. Their `pyproject.toml` declares
   `packages = ["matanyone2"]` *and* force-includes `matanyone2/config`, so the
   config files are added twice. Fixed by removing the redundant force-include.

2. **Their dependency list stalls resolution.** It pulls training and GUI
   packages inference never uses — `tensorboard`, `pycocotools`, `hickle`,
   `thinplate`, `PySide6`, `pyqtdarktheme`, `gradio` — plus `cchardet` and
   `netifaces`, which no longer build on Python 3.10+. Resolved by installing
   only the inference dependencies, then `pip install --no-deps -e`.

Installed in an isolated `venv_ma2/` so the working pipeline stays untouched.
Model: 35.2M params, **runs on MPS** (their `get_default_device()` checks for it).

### API compatibility: essentially a drop-in

| | v1 | v2 |
|---|---|---|
| `InferenceCore(network, cfg, device=)` | ✅ | ✅ same |
| `step(image, mask, objects, first_frame_pred=)` | ✅ | ✅ **identical** |
| `output_prob_to_mask()` | ✅ | ✅ (adds `matting=` kwarg) |
| Import path | `matanyone.inference.inference_core` | `from matanyone2 import InferenceCore` |

The existing warmup loop in `phase2b_matting.py` runs against v2 unchanged.

### Measured: Manvi frames, identical SAM 2 seed mask, 768, MPS

| Metric | v1 | v2 |
|---|---|---|
| **Speed** | 0.340 s/frame | **0.354 s/frame** (4% slower) |
| Coverage | 25.0% | 24.9% |
| **Soft px** | 1.61% | **1.18%** (−27%) |
| Silhouette IoU | — | 99.2% |
| hairMAE vs v1 | — | 28.7 |

### Verdict: no reason to switch for this workload

- **No speed gain** — marginally slower.
- **Visually near-identical.** Side-by-side composites show both preserving the
  same loose strands with clean edges. The 27% drop in soft pixels is a
  *tighter, more confident* matte, not lost detail.
- The paper's gains are on hard real-world benchmarks (occlusion, motion blur,
  crowded scenes). Our clips are well-lit talking heads against plain
  backgrounds — v1 was already near its ceiling here, so there is little for v2
  to improve.

**Initial recommendation was to stay on v1. The full-clip run overturned the
timing half of that — see R5.**

### R5 — MatAnyone 2, full 359-frame run (corrects R4's timing)

The R4 comparison used a 40-frame slice, which was **too short to time fairly**:
model load and the 10 warmup iterations dominated it. Re-measured on the whole
clip, with identical phase 3 / 4 settings so the model is the only variable.

| Phase | v1 | v2 |
|---|---|---|
| 2 matting | 122 s (0.340 s/frame) | **113 s (0.315 s/frame)** |
| 3 composite | 103 s | 103 s |
| 4 render | 3 s | 3 s |

**v2 is ~10% faster on the full clip**, not 4% slower as the slice suggested.

| Quality | v1 | v2 |
|---|---|---|
| Coverage | 25.1–28.0% | 25.0–27.9% |
| Soft px | 1.59–1.68% | 1.17–1.23% |
| Subject shift | 0.40–0.42 | **0.40–0.42** |
| **Edge/subject ratio** | 0.47–0.50 | **0.44–0.46** |

Output: `outputs/final_manvi_ma2.mp4` (7.2 MB)

**Revised verdict: v2 is a modest win.** Slightly faster, slightly better edge
containment, visually equivalent. Not the −26% MAD the paper reports — our
well-lit talking-head footage is not what that benchmark measures — but it is
better on both axes rather than a wash.

Still not urgent to switch: the gain is ~9 s on a 220 s run. The reason to
adopt it is the API is a drop-in and the edge metric is genuinely better, so
switching costs little. Blockers are the two upstream packaging bugs in R4,
which need the documented workarounds.

**Method note:** never benchmark a model on a slice short enough for
fixed startup cost to dominate. The 40-frame result was not wrong arithmetic —
it measured the wrong thing.

---

## R6 — Subject relighting (fixes the cutout look)

User observation: subject and background "both feel cutout". Diagnosed by
measuring, not by eye.

### Why it read as pasted-on

| | Subject | Purple bg |
|---|---|---|
| Brightness L* | 148 | 33 |
| Key light | from the **left** (166 vs 130) | from the **upper-right** |
| Colour | warm beige bounce | purple |

She was **115 L\* brighter than the room she was supposedly standing in**, lit
from the opposite side. Both cues are read instantly by a viewer.

The existing `color_harmonize` passes could not fix this: they operate on
global statistics and edge pixels only, so they tint a silhouette rather than
light a person. That is also why turning harmonization off (R1) was correct —
it was applying a global wash, not lighting.

### New: `relight.py` (operates on the subject's INTERIOR)

| Function | Purpose |
|---|---|
| `estimate_light_direction` | Infers the key direction from background luminance |
| `match_subject_exposure` | LAB gain toward the room's level, damped to keep contrast |
| `apply_ambient_bounce` | Room colour cast, weighted toward shadow regions |
| `apply_directional_light` | Brightens the side facing the key, darkens the other |
| `add_contact_shadow` | Grounds the subject against the background |

### New background: `inputs/white_studio_bg.png`

1080x1920, AR 0.5625. White wall, soft grey shadow falloff, **lit upper-left to
match her own key**. Choosing a background whose light agrees with the subject
cut the gap from 115 L* (purple) to 66 before any relighting at all.

### Settings used and result

`--relight --relight-exposure 0.25 --relight-bounce 0.15
 --relight-directional 0.35 --contact-shadow 0.55`

| | Value |
|---|---|
| Subject/bg gap, no relight | 66.3 L* |
| **Subject/bg gap, relit** | **49.3 L*** |
| Composite + render, 359 frames | 136 s |
| Temporal stability | 0.37 L* mean frame-to-frame, max 1.11 — **no flicker** |

Output: `outputs/final_manvi_relit.mp4` (7.4 MB)

### Two things worth recording

**A near-miss caught by tracing rather than tweaking.** Four tests failed on
first run. Both the test helper's gradient *and* a negation inside
`apply_directional_light` were inverted. Flipping signs until green would have
left a real bug — the code was lighting the side *away* from the key. Tracing
the convention end to end found it.

**Light direction drifts with background motion.** Ken Burns pans the plate, so
the estimated direction moves from dx −0.063 to −0.081 across the clip. Checked
for flicker rather than assuming: 0.37 L* mean change is imperceptible, so it
was left alone. If a future background has a stronger gradient, pin the
direction once and reuse it.

### Honest limitations

- Contact shadow is barely visible against a white background; it needs a
  larger offset, or it mainly helps on darker backdrops.
- This is a 2D approximation. Surface orientation is inferred from mask
  geometry and luminance, not real normals. Depth-based relighting (e.g. Depth
  Anything) would be a genuine step up, and a much larger project.
