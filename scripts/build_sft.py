# -*- coding: utf-8 -*-
"""Build the supervised fine-tuning mixture from the local corpus.

Produces ``data/sft/sft.jsonl`` with five task kinds:

  register   write a passage that opens with a given sentence
  long       450-900 character passage, length-faithful
  landing    a passage that must end on a given sentence
  chain      continue from a two-sentence tail, keeping the scene
  plan       three-beat plan followed by the passage

Each record is ``{"system", "user", "assistant", "kind"}`` in ChatML form and
can be fed directly to ``train_lora_3b.py --data``.

Usage
-----
    python scripts/build_sft.py
    python scripts/build_sft.py --per-kind 400 --out data/sft/sft.jsonl
"""
import argparse, json, random, re

from paths import SFT, corpus_files, read_paras

CJK = re.compile(r"[\u4e00-\u9fff]")
SENT = re.compile(r"(?<=[。！？])")

SYS = ("你是中文小说家，译笔取法旧译：句子沉实，长短参差，白描，只写看得见的动作与物，"
       "不议论，不总结。不用破折号和省略号。风格：俄国旧译。只输出正文。")
SYS_LAND = ("你是中文小说家。先在心中倒推这一段要走到哪一句，再从那件小事起笔。"
            "只写动作、物件、光与声音，不解释心理，不总结。只输出正文。")
SYS_CHAIN = ("你是中文小说家。读下面的上文，让下一段由它的后果推动，不另起话题，不解释。"
             "只写动作、物件、光与声音。只输出正文。")


def cjk(s):
    return len(CJK.findall(s))


def sents(s):
    return [x.strip() for x in SENT.split(s) if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-kind", type=int, default=500)
    ap.add_argument("--min", type=int, default=150)
    ap.add_argument("--max", type=int, default=900)
    ap.add_argument("--out", default=str(SFT / "sft.jsonl"))
    ap.add_argument("--seed", type=int, default=20261112)
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    pool = []
    for path in corpus_files():
        for p in read_paras(path):
            n = cjk(p)
            if args.min <= n <= args.max and len(sents(p)) >= 4:
                pool.append(p)
    if not pool:
        raise SystemExit("Corpus is too small: need paragraphs between "
                         f"{args.min} and {args.max} CJK characters.")
    rnd.shuffle(pool)

    rows = []
    for p in pool:
        ss = sents(p)
        rows.append({"system": SYS, "user": f"写一段小说，以这句话开头：{ss[0][:30]}",
                     "assistant": p, "kind": "register"})
        if cjk(p) >= 450:
            rows.append({"system": SYS, "user": f"写一段小说，{min(800, cjk(p))} 字左右，以这句话开头：{ss[0][:30]}",
                         "assistant": p, "kind": "long"})
        last = ss[-1]
        if 8 <= cjk(last) <= 40:
            rows.append({"system": SYS_LAND,
                         "user": f"【落点】这一段要落在这句话上：\n{last}\n\n写一段小说，最后自然落到这句话。",
                         "assistant": p, "kind": "landing"})
        if len(ss) >= 3:
            rows.append({"system": SYS_CHAIN,
                         "user": f"上文：{''.join(ss[:2])}\n\n接着写这一段，不要重复上文。",
                         "assistant": p, "kind": "chain"})
        if len(ss) >= 5:
            beats = "\n".join(f"这一拍：{s[:20]}" for s in ss[1:4])
            rows.append({"system": "你是小说编辑。只为这一段列出三拍，每拍一行，以“这一拍：”开头，不超过二十字。",
                         "user": f"为下面这一段列三拍（写完后才给出正文）：\n{p[:120]}",
                         "assistant": beats, "kind": "plan"})

    rnd.shuffle(rows)
    by_kind = {}
    out_rows = []
    for r in rows:
        k = r["kind"]
        if by_kind.get(k, 0) < args.per_kind:
            by_kind[k] = by_kind.get(k, 0) + 1
            out_rows.append(r)

    from pathlib import Path
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out}: {len(out_rows)} examples {by_kind}")


if __name__ == "__main__":
    main()
