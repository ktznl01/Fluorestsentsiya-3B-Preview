# Model Card — Fluorestsentsiya-3B-Preview v0.2

## Model details

| Item | Value |
|---|---|
| Model | Fluorestsentsiya-3B-Preview v0.2 |
| Base | Qwen2.5-3B-Instruct (3.09B) |
| Adapter | LoRA r=16, α=32, dropout=0.05, 29.9M parameters |
| Quantization | base GGUF Q4_K_M + LoRA GGUF f16 |
| Inference | llama.cpp b12018 + Vulkan; 8 CPU threads; context 8192 |
| License | Non-commercial research only (model); base Apache-2.0 |

## Intended use

- Generate Chinese narrative drafts in an old-translation register.
- Continue a human opening toward 1,500–2,000 Chinese characters.
- Serve as a draft generator before human revision.

## Out-of-scope use

- Do not present outputs as human-authored or as “human-level” literature.
- Do not use commercially; do not relicense the adapter or base.
- Do not reuse the contaminated CWB 4.36 number from the parent project.
- Do not assume 21/21 prompts reach 2,000 characters.

## Training data

- User-provided Chinese translations of Chekhov and Dostoevsky; corpora are not redistributed.
- 248 curated 2k continuation examples; 1,200 sliding-window examples (experimental).
- 12 offline teacher continuations used only as training data; the teacher is not loaded at inference.
- Base v0.1 register adapter was the initialization; v0.2 continues that adapter.

## Evaluation

21 held-out prompts, model-only outputs, no human rewrite:

| Version | Mean CJK | 2k hit | rep4 | Duplicate sentence | distinct4 |
|---|---:|---:|---:|---:|---:|
| Base Qwen2.5-3B | 1,120.2 | 1/21 | 0.0762 | 0.00% | 0.9238 |
| v0.1 | 830.1 | 4/21 | 0.2503 | 17.32% | 0.7497 |
| v0.2 | 1,497.7 | 11/21 | 0.1827 | 2.70% | 0.8173 |

Metric definitions and the full per-sample table are in [`bench/v0.2/README.md`](../bench/v0.2/README.md). The chart is `bench/v0.2/final21_bar_corporate.png`.

## Limitations

- Only 11/21 samples reach 1,800+ characters; the mean is 1,498.
- distinct4 improves by 9.0% over v0.1, not 20%; the human target is 0.9534, only +14.9% over v0.1.
- The base model has better rep4/distinct4 but writes much shorter outputs (1,120 mean, 1/21 at 2k); the comparison is not length-matched.
- Role and causal consistency were not blind-rated at n=21.
- CWB/Lech/POLLUX/WPB/Burrows were not re-run for v0.2; v0.1's original benchmark numbers are kept separately in `docs/legacy-v0.1.md`.
- The model can still drift into memorized passages at long lengths.

## Ethical and licensing notes

The model imitates a prose register, not a living author. Training corpora are user-supplied and not redistributed. Model weights and outputs are for non-commercial research; see `LICENSE`. Base model remains Apache-2.0.
