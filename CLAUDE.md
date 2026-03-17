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

## Architecture — 4-Phase Pipeline

All phases are orchestrated by `run_pipeline.py`, which calls into each module in sequence:

1. **phase1_extract.py** — Extracts every frame from the video as PNG files using OpenCV. Also has `rebuild_video()` for verifying lossless round-tripping.

2. **phase2_segment.py** — Runs SAM 2 video segmentation. Key design decisions:
   - **Downscales** frames (default max 1024px long side) before feeding to SAM 2 to fit in GPU memory.
   - **Chunks** the frame sequence (default 450-500 frames/chunk) to avoid MPS memory limits. Each chunk gets its own SAM 2 predictor instance, prompted on frame 0.
   - **Upscales** output masks back to original resolution.
   - Prompt points provided at original resolution are automatically scaled down for SAM 2.

3. **phase3_composite.py** — Alpha-blends foreground (original frame * mask) onto the new background. Feathers mask edges with Gaussian blur (`--blur-radius`, default 5).

4. **phase4_render.py** — Assembles composited frames into MP4 with OpenCV, then merges audio from the original video using ffmpeg.

## Directory Layout (Runtime)

- `inputs/` — Source video and background images
- `outputs/frames/` — Extracted original frames (`frame_NNNNNN.png`)
- `outputs/frames_scaled/` — Downscaled frames for SAM 2 (with `chunk_*` subdirs)
- `outputs/masks/` — Binary segmentation masks (`mask_NNNNNN.png`)
- `outputs/composited/` — Final composited frames (`comp_NNNNNN.png`)
- `outputs/final.mp4` — Final output with audio
- `checkpoints/` — SAM 2 model weights (`sam2.1_hiera_large.pt`)

## Known Issue

`run_pipeline.py` imports `segment_video` and `show_first_frame_for_prompt` from `phase2_segment`, but that module defines `run_segmentation` instead. The standalone phase2 works, but the orchestrator may fail on import.

## Key Dependencies

- `sam-2` (Meta's SAM 2), `torch`, `torchvision`, `opencv-python`, `numpy`, `pillow`
- `ffmpeg` (system binary, not a Python package) — needed for audio merging only
