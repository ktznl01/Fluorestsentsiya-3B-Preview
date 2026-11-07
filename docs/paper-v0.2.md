# Fluorestsentsiya-3B-Preview v0.2: A 3B Chinese Narrative Model at the 2,000-Character Frontier

**Technical report — 2026-11-07**

## Abstract

We present v0.2 of Fluorestsentsiya-3B-Preview, a 3.09B-parameter Chinese narrative model built by continuing a rank-16 LoRA adapter on Qwen2.5-3B-Instruct. The release targets a practical but narrow goal: generate a 1,500–2,000-character Chinese short-story draft in the register of nineteenth-century Russian fiction in Chinese translation. v0.2 is trained on 248 curated 2k continuation examples, 1,200 sliding-window examples, and 12 offline teacher continuations; the teacher is used only for data construction and is never loaded at inference. On 21 held-out prompts, with all outputs produced by the model itself and trimmed at 2,000 Chinese characters, v0.2 reaches a mean of 1,497.7 characters, 11/21 outputs at 1,800+ characters, rep4 0.1827, duplicate-sentence rate 2.70%, and distinct4 0.8173. The v0.1 baseline reaches 830.1 characters, 4/21 at 2k, rep4 0.2503, duplicate-sentence rate 17.32%, and distinct4 0.7497; the base Qwen2.5-3B-Instruct reaches 1,120.2 characters, 1/21 at 2k, rep4 0.0762, duplicate-sentence rate 0.00%, and distinct4 0.9238. v0.2 therefore improves length and repetition over v0.1, but does not reach 21/21 at 2k, does not improve distinct4 by 20%, and is not a human-level writer. We report the negative results and the metric ceiling explicitly.

## 1. Introduction

Small local models are attractive for Chinese narrative writing because they can run on consumer hardware, but they are weak at long-form structure. In our earlier v0.1 release, a 3B model with a 29.9M-parameter adapter could imitate an old-translation register at short lengths, yet a single call typically stopped at 200–500 characters. Segment-and-concatenate pipelines could reach 2,000 characters, but only at the cost of repetition and scene drift. The practical question for v0.2 was therefore narrow: can a 3B model, without a 7B generator or an API at inference, produce a usable 1,500–2,000-character draft with lower repetition than v0.1?

We answer with a measured “partly”. v0.2 increases mean length by 80.4% over v0.1 and reduces rep4 by 27.0%, but only 11/21 samples reach 1,800 characters, and the distinct4 gain is 9.0%, below the 20% target. The base model has lower rep4 and higher distinct4, but writes much shorter outputs; its apparent advantage is largely a length effect. We keep the base comparison in the paper because it is the honest reference, not because it satisfies the task.

Our contributions are:

1. a v0.2 adapter trained by continuing the v0.1 register adapter on human 2k continuations plus offline teacher revisions;
2. a 21-prompt model-only evaluation protocol that trims every output at 2,000 Chinese characters on a sentence boundary;
3. an explicit separation of base, v0.1, and v0.2 results, with v0.1's legacy benchmark numbers kept apart from the new 21-sample table;
4. a documented negative result: at 3B scale, length, repetition, and lexical diversity do not improve monotonically with more teacher data.

## 2. Related Work

**Long-output SFT.** LongWriter and related work show that supervised target length is a first-order constraint on output length: models trained on short targets rarely extrapolate to much longer outputs. We therefore build 2k continuation targets rather than 512-token snippets.

**Plan-and-write and revision.** DOC, Re3, and SuperWriter decompose long writing into plans, drafts, and revisions. We tested scene pipelines and two-pass continuation; both can increase length, but at 3B they often increase repetition or drift. v0.2 keeps a single-call interface and only trims at the 2k boundary.

**Preference optimization.** ORPO removes the reference model and combines SFT with an odds-ratio term. In our setting, ORPO with human/teacher chosen and model-drafted rejected produced a negative reward gap: the model assigned higher likelihood to its own short/looped drafts. We therefore do not claim an ORPO gain in v0.2.

**Decoding-time repetition control.** DRY and DRY-C penalize repeated n-grams without a fixed repeat window. DRY is necessary for v0.2: without it, the adapter produces paragraph-level loops at long lengths. DRY reduces exact repetition but does not by itself produce 2k closure.

**Evaluation.** We use surface metrics (length, rep4, duplicate sentence rate, distinct4) plus a qualitative reading. These metrics are necessary but not sufficient: a model can score well by writing a short, clean text, or by drifting into a different story. We report both the numbers and the failure modes.

## 3. Model and Data

### 3.1 Base and adapter

The base is Qwen2.5-3B-Instruct: 3.09B parameters, 36 layers, hidden size 2048, GQA, SwiGLU, RoPE, vocabulary 151,936, native context 32,768. The adapter is LoRA with rank 16, alpha 32, dropout 0.05, applied to `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` in all layers; 29.9M trainable parameters (0.97%). v0.2 continues the v0.1 register adapter rather than starting from the base.

### 3.2 Data

The corpus is user-supplied Chinese translations of Chekhov and Dostoevsky; it is not redistributed. We construct:

- 248 curated 2k continuation examples from complete stories and chapter windows;
- 1,200 sliding-window examples (1,800–2,200 Chinese characters per window) used in earlier experiments;
- 12 offline teacher continuations (1,441–1,746 characters) written by the assistant and used only as training data.

The 12 teacher examples are mixed with 65 human 2k examples and upweighted 5x for the final v0.2 SFT run. The teacher is not present in the released inference path.

### 3.3 Training

We continue the v0.1 adapter for 25 steps at sequence length 2,048, learning rate 1e-5, AdamW, bf16, gradient checkpointing, 8 CPU threads. The loss is standard next-token cross-entropy on the assistant span; the prompt is masked. The final checkpoint is step 15, selected by held-out surface metrics.

## 4. Evaluation Protocol

### 4.1 Prompts and generation

We use 21 held-out openings from the user corpus. For each prompt, we generate one continuation with `max_tokens=2400`. The base and v0.1 use `temperature=0.7`, `top_p=0.9`, `repeat_penalty=1.1`; v0.2 uses `temperature=0.7`, `top_p=0.9`, `repeat_penalty=1.15`, and DRY with multiplier 0.8, base 1.75, allowed length 2, and penalty last-n 2,048. All outputs are trimmed at 2,000 Chinese characters on the last sentence boundary before the limit. No human rewrite is used.

### 4.2 Metrics

- **Mean CJK characters**: mean number of Chinese characters after trimming.
- **2k hit**: fraction of samples with at least 1,800 Chinese characters after trimming.
- **rep4**: 4-gram repetition rate, (total 4-grams − unique 4-grams) / total 4-grams. Lower indicates less repetition.
- **Duplicate sentence rate**: exactly repeated sentences / total sentences. Lower indicates fewer repeated sentences.
- **distinct4**: unique 4-grams / total 4-grams. Higher indicates greater 4-gram diversity.
- **Relative change**: ((v0.2 − v0.1) / v0.1) × 100%. For repetition metrics, a negative value is an improvement.

We do not report a composite score, because the metrics trade off and a single scalar would hide the length/repetition failure modes.

## 5. Results

### 5.1 Main table

| Version | Mean CJK | 2k hit | rep4 | Duplicate sentence | distinct4 |
|---|---:|---:|---:|---:|---:|
| Base Qwen2.5-3B | 1,120.2 | 1/21 | 0.0762 | 0.00% | 0.9238 |
| v0.1 | 830.1 | 4/21 | 0.2503 | 17.32% | 0.7497 |
| v0.2 | 1,497.7 | 11/21 | 0.1827 | 2.70% | 0.8173 |

v0.2 vs v0.1: mean CJK +80.4%; rep4 −27.0%; duplicate sentence −84.4%; distinct4 +9.0%.

### 5.2 Figure

The 21-sample comparison is in `bench/v0.2/final21_bar_corporate.png`. The chart is generated deterministically from `bench/v0.2/final21_metrics.json`; no image-generation model is used for the numbers.

### 5.3 What improved

v0.2 is a clear improvement over v0.1 on length (830 → 1,498 mean; 4/21 → 11/21 at 2k) and on exact repetition (17.32% → 2.70% duplicate sentences; rep4 0.250 → 0.183). The improvement is consistent with the training change: 2k human targets plus a small set of teacher continuations, with DRY at inference.

### 5.4 What did not improve

- 11/21 is not 21/21. Ten prompts still stop below 1,800 characters.
- distinct4 improves by 9.0%, not 20%. The human target on the same 21 prompts is 0.9534, only +14.9% over v0.1; the metric is close to its ceiling.
- The base model has rep4 0.0762 and distinct4 0.9238, better than v0.2, but writes only 1,120 characters and reaches 2k in 1/21. A length-matched comparison is not possible because the base rarely reaches the target length.
- Role and causal consistency were not blind-rated at n=21; we do not claim a quantitative gain there.

## 6. Analysis

### 6.1 Length is still the first bottleneck

The 2k hit rate is the clearest summary: 11/21. The distribution is bimodal — some prompts stop at 500–900 characters, others overshoot 2,800 and are trimmed. The model has learned that 2k is possible, but not a stable stopping policy. A single-call interface is therefore not sufficient for a 21/21 guarantee.

### 6.2 DRY is necessary but not sufficient

Without DRY, v0.2 produces paragraph-level loops at long lengths. With DRY, exact repetition drops sharply, but the model can still drift into memorized passages or repeat a scene at a semantic level. surface rep4 does not capture semantic loop.

### 6.3 Teacher data is not a monotonic lever

Across our internal runs, 4, 6, 8, 10, and 12 teacher segments did not produce monotonic gains. The 12-segment run reached 2,493 mean characters at step 15 but collapsed on one prompt to 3,058 characters with rep4 0.450; a 5-sample trim reduced the mean to 1,865 and still left p3 at rep4 0.341. This is why v0.2 is a measured release, not a claim of human-level writing.

### 6.4 Metric ceiling

distinct4 cannot be improved by 20% on this data: v0.1 0.8241 × 1.2 = 0.989, while the human target is 0.9396 on the same three prompts and 0.9534 on the 21. Even human-written targets do not clear a literal +20% rule. We therefore report the raw value and the ceiling rather than a guaranteed relative gain.

## 7. Limitations

1. **Scale**: 21 prompts is a small evaluation; confidence intervals are wide.
2. **Length**: 11/21 at 2k; no 21/21 guarantee.
3. **Repetition**: exact loops are reduced, semantic loops and cross-story drift remain.
4. **Character/causal consistency**: not blind-rated at n=21.
5. **Benchmarks**: Lech/POLLUX/WPB/Burrows were not re-run for v0.2. v0.1's original numbers are preserved in `docs/legacy-v0.1.md` and must not be mixed with the v0.2 21-sample table.
6. **CWB**: the parent-project CWB 4.36 is contaminated by 10/12 prompts and is not reused.
7. **License**: research-only; corpora are not redistributed.

## 8. Conclusion

v0.2 is a real but partial improvement: longer outputs, much lower exact repetition, and a measurable but sub-20% gain in lexical diversity. It is not a human-level model and does not satisfy a literal “all metrics +20%” rule. The honest release is a 3B draft generator with a 21-sample model-only evaluation, a chart, a model card, and an explicit list of failures.

## References

1. Qwen Team. Qwen2.5 Technical Report. arXiv:2412.15115.
2. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
3. Hong et al. ORPO: Monolithic Preference Optimization without Reference Model. EMNLP 2024.
4. LongWriter: Unleashing 10,000+ Word Generation from Long Context LLMs. arXiv:2408.07055.
5. DOC: Detailed Outline Control for Long-form Generation. arXiv:2406.01388.
6. Re3: Plan, Reprompt, Rerank, Edit. arXiv:2210.06774.
7. SuperWriter: Reflection-Driven Long-Form Generation. arXiv:2506.04180.
8. DRY: Don't Repeat Yourself: Preventing Repetition in LLM Generation. arXiv:2305.14688.
9. Lech Mazur. LLM Creative Story-Writing Benchmark. GitHub: lechmazur/writing.
10. POLLUX: Eye of Judgement. arXiv:2505.24616.
11. Burrows. Delta: a measure of stylistic difference. Literary and Linguistic Computing, 2002.

## Appendix A. Reproduction

See `README.md` and `bench/v0.2/README.md`. The v0.2 adapter is `adapters/fluorestsentsiya-3b-preview-v0.2/`; the LoRA GGUF is `gguf/fluorestsentsiya-3b-preview-v0.2-lora-f16.gguf`. The base GGUF is `gguf/fluorestsentsiya-3b-preview-Q4_K_M.gguf`. The evaluation script and chart are in `bench/v0.2/`.
