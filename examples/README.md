# Examples: v0.1 vs v0.2

The three v0.1 examples are preserved as historical reference files. They are the original repository examples, renamed with `_v0.1` in the filename; their prose and headers are otherwise unchanged.

The three v0.2 files were generated once from the **same prompts and target lengths** with the released v0.2 base + LoRA, so readers can compare the two versions side by side. This is an illustrative comparison, not a controlled experiment: the v0.1 files were produced by an earlier multi-pass/edited pipeline, while the v0.2 files are single-call generations.

## Files and surface metrics

| Prompt | v0.1 file | v0.1 CJK | v0.1 rep4 | v0.1 dup sentence | v0.1 distinct4 | v0.2 file | v0.2 CJK | v0.2 rep4 | v0.2 dup sentence | v0.2 distinct4 |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 入冬 / 500 | `01_winter_parting_500_v0.1.md` | 581 | 0.2270 | 15.00% | 0.7730 | `01_winter_parting_500_v0.2.md` | 308 | 0.0164 | 0.00% | 0.9836 |
| 药与酒 / 1,000 | `02_illness_alcohol_1000_v0.1.md` | 971 | 0.1791 | 8.77% | 0.8209 | `02_illness_alcohol_1000_v0.2.md` | 432 | 0.0777 | 0.00% | 0.9223 |
| 空下来的手 / 2,000 | `03_unemployment_love_2000_v0.1.md` | 1,837 | 0.4974 | 37.04% | 0.5026 | `03_unemployment_love_2000_v0.2.md` | 2,024 | 0.1212 | 0.00% | 0.8788 |

Reading the table: v0.2 has much lower exact repetition on all three prompts, but on the 500- and 1,000-character prompts it stops far below the target length. Only the 2,000-character prompt reaches the target. The v0.1 files are longer on the first two prompts because they came from a different multi-pass/editing pipeline; length is therefore not a clean model-to-model comparison.

Metric definitions are the same as `bench/v0.2/README.md`: rep4 is the 4-gram repetition rate, duplicate sentence is the fraction of exactly repeated sentences, and distinct4 is unique 4-grams / total 4-grams. The metrics are surface diagnostics, not literary judgments.

## Exact prompts used for both versions

### 01 — 入冬 / 500

```text
写一篇约五百字的小说。入冬第一场雪，一个女人来向他道别；他们曾经相爱，现在她要跟别人走。只写动作、物件、天气，不写心理，不总结。
```

### 02 — 药与酒 / 1,000

```text
写一篇约一千字的小说。一个男人病了，医生让他戒酒；他把酒瓶藏进柴房，夜里起来又去喝。写他一天里的几件事：看病、买药、藏酒、被妻子发现。只写动作与物，不解释。
```

### 03 — 空下来的手 / 2,000

```text
写一篇约两千字的小说。工厂停工，他失了业；妻子在洗衣房做工，早出晚归。入冬了，屋里冷，他每天出门假装去找活，其实在车站坐到天黑。写他这些天里的几件事：出门、车站、遇见旧工友、回家、与妻子吃饭。只写动作、物件、天气与对话，不写心理，不总结。
```

Both versions use the same system prompt:

```text
你是中文小说家，译笔取法旧译：句子沉实，长短参差，白描，只写看得见的动作与物，不议论，不总结。不用破折号和省略号。风格：俄国旧译。只输出正文。
```

## v0.2 generation settings

- Base: `gguf/qwen25-3b-base-Q4_K_M.gguf`
- LoRA: `gguf/fluorestsentsiya-3b-preview-v0.2-lora-f16.gguf`
- `temperature=0.7`, `top_p=0.9`, `repeat_penalty=1.15`
- DRY: `multiplier=0.8`, `base=1.75`, `allowed_length=2`, `penalty_last_n=2048`
- `max_tokens = target × 2.2`; outputs trimmed at `target × 1.05` on a sentence boundary
- One generation per prompt; no reranking, no human rewrite, no teacher call at inference

The v0.1 files retain their original headers and sampling notes. They were not re-run with v0.2.
