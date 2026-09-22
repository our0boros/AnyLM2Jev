# Agent Instructions

## Project Direction

- The research question is narrow: **can a self-consistency-distilled readout head
  beat raw option-logit readout** for Jev-style decision queries on a small frozen
  LM?
- Treat `p_raw` (frozen LM label logits, one forward pass) as the baseline that any
  claimed method must beat. `q*` (paraphrase x order aggregation, V forward passes)
  is the ceiling a single-pass head is trying to compress. Report all three.
- The backbone stays frozen. Only the head (and, behind a flag, LoRA) is trained. A
  head on frozen features cannot exceed what those features encode; do not claim
  otherwise. The defensible wins are calibration, order invariance and format
  decoupling, measured with ECE / Brier / NLL / coverage-at-risk.
- Never train the head with hard labels. Targets are the soft aggregated `q*`.
  Hard-label CE throws away exactly the confidence signal this project is about.
- **Read option hidden states at the end of the option text, never at the option's
  label letter.** A causal LM at the label position has not read the option yet, so
  a head reading there scores text it cannot see and silently does nothing.
- **Debias the target, not the head.** The label-free prior correction must be
  folded into `q*` before distillation. Applying it to an already-trained head
  double-corrects and collapses it.
- Results and the bugs that mattered: `docs/RESULTS.md`. Do not overstate accuracy;
  report the paired CI and the coverage@5% number alongside it.

## Layout And Commands

- Reusable code lives in `src/jev/`; runnable entrypoints in `scripts/`; deterministic
  tests in `tests/`; generated artefacts in the ignored `data/`, `runs/`, `artifacts/`.
- The package is not installed. Run with `PYTHONPATH=src`.
- Everything runs on a single GPU; `run.sh` drives the whole pipeline
  (`data` / `wanli` / `banking77` / `banking77_full` / `figures` / `all`).
- Keep `torch`/`transformers` imports out of pure data/schema code so those modules
  stay testable without a GPU environment.

## Changes

- Prefer the smallest change that advances the `p_raw` vs `p_head` comparison.
- Keep experiment knobs as CLI flags, not hard-coded constants, so runs are
  reproducible.
- Do not commit or delete unrelated worktree changes unless explicitly requested.
