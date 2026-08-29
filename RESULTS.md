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
