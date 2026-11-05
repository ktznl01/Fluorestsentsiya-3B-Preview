# -*- coding: utf-8 -*-
"""Build the domain-adaptive pre-training corpus.

Reads every ``data/corpus/*.txt`` file (UTF-8, paragraphs separated by blank
lines), filters boilerplate and over-short/long paragraphs, de-duplicates and
writes one contiguous corpus to ``data/processed/corpus.txt``.

Usage
-----
    python scripts/build_corpus.py
    python scripts/build_corpus.py --min 80 --max 900 --out data/processed/corpus.txt
"""
import argparse, random, re

from paths import PROCESSED, corpus_files, read_paras

CJK = re.compile(r"[\u4e00-\u9fff]")
BOILER = ("版权", "目录", "http", "译者", "责任编辑", "年谱", "作品集", "注释", "Z-Library")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=80, help="minimum CJK characters per paragraph")
    ap.add_argument("--max", type=int, default=900, help="maximum CJK characters per paragraph")
    ap.add_argument("--out", default=str(PROCESSED / "corpus.txt"))
    ap.add_argument("--seed", type=int, default=20261111)
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    paras, seen = [], set()
    for path in corpus_files():
        for p in read_paras(path):
            n = len(CJK.findall(p))
            if n < args.min or n > args.max:
                continue
            if any(b in p for b in BOILER):
                continue
            key = re.sub(r"\s+", "", p)[:40]
            if key in seen:
                continue
            seen.add(key)
            paras.append(p)
    rnd.shuffle(paras)

    from pathlib import Path
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(paras) + "\n", encoding="utf-8")
    chars = sum(len(CJK.findall(p)) for p in paras)
    print(f"wrote {out}: {len(paras)} paragraphs, {chars} CJK characters")


if __name__ == "__main__":
    main()
