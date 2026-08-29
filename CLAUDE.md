# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Philosophy — How Claude Must Work in This Repo

You are a staff-level engineer. Act like one. Every decision should reflect deep experience and disciplined thinking.

### Zero Assumptions — Ask First, Build Second

- **Never assume requirements, edge cases, or user intent.** If something is ambiguous or underspecified, stop and ask the user before writing a single line of code.
- Ask all clarifying questions upfront in one go — don't drip-feed questions across multiple rounds.
- Questions should cover: scope, expected behavior, error handling strategy, performance constraints, compatibility requirements, and how the change fits into the broader system.
- Only after every question is answered should you move to planning.

### Plan Thoroughly in Phases Before Coding

- Before any implementation, produce a clear phased plan and present it to the user for approval.
- Each phase should be a self-contained unit of work with a clear deliverable.
- The plan must identify: which files are created/modified, what each phase accomplishes, dependencies between phases, and what "done" looks like for each phase.
- Wait for explicit user approval before starting implementation.
- Implement one phase at a time. Confirm completion of each phase before moving to the next.

### Test-First, Verify-Everything Development

This is non-negotiable. A staff engineer writes the test before the implementation — always.

- **Tests come first.** Before writing any feature or fix, write a failing test that defines the expected behavior. Then make it pass. This is the workflow, every single time.
- **Verify after every change.** After implementing anything — a function, a refactor, a bug fix — run the relevant tests immediately. Do not move on until they pass. Do not batch verification for later.
- **Actively try to break your own code.** After tests pass, think adversarially: What inputs would break this? What happens at boundaries (empty list, zero frames, missing file, corrupt image, single-frame video)? What if the filesystem is full? What if ffmpeg isn't installed? Write tests for those cases.
- **Hunt for loose ends and close them.** After each phase of work, do a thorough audit:
  - Are there any unhandled error paths?
  - Are there any new functions without tests?
  - Are there any TODOs or placeholders left behind?
  - Does the change break any existing contract or interface?
  - Are there any resource leaks (unclosed files, unreleased video captures, GPU memory not freed)?
  - Never leave a loose end. If you find one, fix it before moving on.
- **Verification is continuous, not a final step.** Don't treat testing as a phase at the end. It's woven into every step — write test, implement, verify, stress-test, audit, then move forward.
- **Run the full relevant test suite before declaring any phase complete.** Regressions are unacceptable. If a change in phase 3 breaks phase 1, that's your problem to fix before proceeding.

### Aggressive Parallelism — Use Agents, Protect Context

Your context window is precious. Don't waste it on work that can be delegated.

- **Parallelize everything that is independent.** If tasks don't depend on each other's results, launch them as parallel agents in a single message. Never run independent tasks sequentially.
- **Delegate to subagents for any work where you only need the outcome.** Research, code exploration, file searches across many files, test execution, code review — if you only care about the final answer, offload it to an agent. Don't pollute your main context with intermediate search results, large file contents, or verbose test output.
- **Use the right agent type for the job:**
  - `Explore` agents for codebase research and finding files/patterns.
  - `feature-dev:code-explorer` for deep tracing of execution paths and understanding existing features.
  - `feature-dev:code-architect` for designing feature architectures based on existing patterns.
  - `feature-dev:code-reviewer` for reviewing code for bugs, security issues, and quality.
  - `general-purpose` agents for multi-step tasks like running tests, complex searches, or any composite work.
  - `Plan` agents for designing implementation strategies.
- **Run background agents when you have other work to do.** If you can continue with independent work while an agent finishes, use `run_in_background: true`. Don't block waiting for results you don't immediately need.
- **Never duplicate agent work in the main context.** If you delegate a search to an agent, don't also search yourself. Trust the agent's result.
- **Batch parallel agent launches in a single message.** Don't send one agent, wait, send another. If you know you need three independent pieces of information, launch all three agents at once.
- **Use agents for verification too.** After implementing a phase, launch a code-reviewer agent and a test-runner agent in parallel — one reviews the code while the other runs the tests.

### Engineering Principles

- **DRY (Don't Repeat Yourself):** Extract shared logic into reusable functions/modules. If you see duplication, refactor it. Never copy-paste code across files.
- **Single Responsibility:** Each function and module does one thing well. If a function is doing multiple unrelated things, split it.
- **Explicit over implicit:** Clear parameter names, clear return types, clear error messages. No magic values, no hidden side effects.
- **Fail fast and loud:** Validate inputs at boundaries. Raise clear exceptions with actionable messages — never silently swallow errors or return ambiguous defaults.
- **Backward compatibility:** When modifying existing interfaces (function signatures, CLI args, file formats), preserve backward compatibility unless the user explicitly agrees to a breaking change.
- **Minimal blast radius:** Make the smallest change that solves the problem. Don't refactor unrelated code, don't "improve" things that weren't asked about.
- **Test your assumptions:** If the fix depends on a hypothesis (e.g., "this variable is None here"), verify it before building on it.

## What This Project Does

Video background replacement pipeline using Meta's SAM 2 (Segment Anything Model 2). Takes an input video and a background image, segments the foreground subject, and composites it onto the new background.

## Setup

```bash
bash setup.sh          # Creates venv, installs deps, downloads SAM 2.1 checkpoint (~2.4GB)
source venv/bin/activate
```

Requires Python 3.10+, ffmpeg (for audio), and a GPU (MPS on Mac, CUDA on Linux/Windows). Falls back to CPU.

## Running the Pipeline

```bash
# Interactive mode (opens first frame for clicking on the subject):
python run_pipeline.py --video inputs/video.mp4 --background inputs/bg.jpg

# Non-interactive (provide foreground click coordinates):
python run_pipeline.py --video inputs/video.mp4 --background inputs/bg.jpg --points "320,240"

# Multiple foreground points:
python run_pipeline.py --video inputs/video.mp4 --background inputs/bg.jpg --points "320,240,400,300"
```

Each phase can also be run standalone (see `if __name__ == "__main__"` in each file).

## Architecture — 5-Phase Pipeline

All phases are orchestrated by `run_pipeline.py`, which calls into each module in sequence:

1. **phase1_extract.py** — Extracts every frame from the video as PNG files using OpenCV. Also has `rebuild_video()` for verifying lossless round-tripping.

2. **phase2_segment.py** — (Legacy) Runs SAM 2 video segmentation with chunking. Produces binary masks. Used when `--matting-mode sam2` is specified.

2b. **phase2b_matting.py** — (Default) MatAnyone video matting. Uses SAM 2 for first-frame mask only, then MatAnyone propagates soft alpha mattes (0-255 continuous) across all frames. Key advantages over phase2:
   - **Regression vs classification** — produces continuous alpha [0,1] instead of binary 0/1
   - **Hair-level detail** — preserves semi-transparent edges, hair strands, motion blur
   - **Temporal consistency** — memory-based propagation eliminates per-frame flicker
   - **No chunking needed** — MatAnyone handles long videos with its own memory management
   - Requires `checkpoints/matanyone.pth` (~141MB, auto-downloaded from HuggingFace)

3. **phase3_composite.py** — Enhanced compositing pipeline that orchestrates:
   - **Edge-aware mask refinement** via `edge_refine.py` (guided filter, morphological cleanup, distance-based gradient)
   - **Color spill decontamination** via `edge_refine.py` (removes original background bleeding in LAB space)
   - **Color/lighting harmonization** via `color_harmonize.py` (histogram matching, white balance, exposure, ambient cast)
   - **Background movement** via `bg_motion.py` (Ken Burns, parallax drift, depth parallax, motion blur)
   - All features enabled by default, individually togglable via `--no-*` CLI flags.

4. **phase4_render.py** — Assembles composited frames into MP4 using ffmpeg H.264 (with OpenCV mp4v fallback), then merges audio from the original video.

### Enhancement Modules

- **`edge_refine.py`** — Edge-aware mask refinement: morphological erosion, guided/bilateral filtering, cosine-ramp boundary gradient, LAB color decontamination.
- **`color_harmonize.py`** — Lighting harmonization: LAB histogram matching, white balance alignment, exposure correction, ambient color cast on edges.
- **`bg_motion.py`** — Background movement engine: Ken Burns zoom+pan, sinusoidal parallax, depth-based counter-parallax (tracks subject COM), directional motion blur.

## Directory Layout (Runtime)

- `inputs/` — Source video and background images
- `outputs/frames/` — Extracted original frames (`frame_NNNNNN.png`)
- `outputs/frames_scaled/` — Downscaled frames for SAM 2 (with `chunk_*` subdirs)
- `outputs/masks/` — Segmentation masks with soft edges (`mask_NNNNNN.png`)
- `outputs/composited/` — Final composited frames (`comp_NNNNNN.png`)
- `outputs/final.mp4` — Final output with audio
- `checkpoints/` — Model weights: `sam2.1_hiera_large.pt` (SAM 2, ~898MB), `matanyone.pth` (MatAnyone, ~141MB)

## Key Dependencies

- `sam-2` (Meta's SAM 2), `matanyone` (MatAnyone video matting), `torch`, `torchvision`, `opencv-python`, `opencv-contrib-python`, `numpy`, `pillow`, `scipy`
- `ffmpeg` (system binary, not a Python package) — needed for H.264 rendering and audio merging
