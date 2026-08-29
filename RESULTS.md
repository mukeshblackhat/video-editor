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
