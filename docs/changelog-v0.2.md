# Changelog

## v0.2 — 2026-11-07

### Added

- v0.2 PEFT adapter (`adapters/fluorestsentsiya-3b-preview-v0.2/`) trained by continuing the v0.1 register adapter on 2k human-continuation data with offline teacher revisions.
- v0.2 LoRA GGUF (`gguf/fluorestsentsiya-3b-preview-v0.2-lora-f16.gguf`) for base GGUF + `--lora`.
- 21-sample model-only evaluation (`bench/v0.2/`), including a Corporate-style chart and per-sample metrics.
- Three v0.1 reference examples preserved as `examples/*_v0.1.md`, plus v0.2 regenerations from the same prompts as `examples/*_v0.2.md` with a comparison table in `examples/README.md`.
- Full paper-style report (`docs/paper-v0.2.md`) and model card (`docs/model-card-v0.2.md`).
- Legacy v0.1 README preserved as `docs/legacy-v0.1.md`.

### Results

- v0.2: 1,497.7 mean CJK, 11/21 at 2k, rep4 0.1827, duplicate sentence 2.70%, distinct4 0.8173.
- v0.1: 830.1 mean CJK, 4/21 at 2k, rep4 0.2503, duplicate sentence 17.32%, distinct4 0.7497.
- Base: 1,120.2 mean CJK, 1/21 at 2k, rep4 0.0762, duplicate sentence 0.00%, distinct4 0.9238.
- v0.2 vs v0.1: +80.4% mean CJK, −27.0% rep4, −84.4% duplicate sentence, +9.0% distinct4.

### Notes

- v0.2 is not a human-level model; 11/21 reach 1,800+ characters.
- distinct4 is a capped metric: the human target is 0.9534, only +14.9% over v0.1.
- CWB 4.36 is contaminated and is not reused.
- v0.1's Lech/POLLUX/Burrows numbers are preserved in the legacy report; they were not re-run for v0.2.

## v0.1 — 2026-11-05

Initial release: 3B register adapter, Q4_K_M GGUF, Lech/POLLUX/Burrows evaluation, examples, training scripts.
