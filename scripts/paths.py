# -*- coding: utf-8 -*-
"""Repository-relative paths.

Everything the scripts need is resolved from the repository root, so the
project can be cloned and run anywhere. Heavy inputs (corpora, training
outputs, benchmark results) live outside git.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("FL_DATA", ROOT / "data"))
CORPUS = Path(os.environ.get("FL_CORPUS", DATA / "corpus"))
SFT = DATA / "sft"
PROCESSED = DATA / "processed"
TRAIN = ROOT / "train"
BENCH = ROOT / "bench"
QUALITY = BENCH / "quality"

for _d in (DATA, CORPUS, SFT, PROCESSED, BENCH, QUALITY, TRAIN):
    _d.mkdir(parents=True, exist_ok=True)


def corpus_files():
    """Plain-text corpora, one paragraph per blank-line-separated block."""
    files = sorted(CORPUS.glob("*.txt"))
    if not files:
        raise SystemExit(
            f"No corpus found in {CORPUS}.\n"
            "Put one or more UTF-8 .txt files there, paragraphs separated by blank lines, "
            "then re-run this script."
        )
    return files


def read_paras(path, skip=0):
    text = Path(path).read_text(encoding="utf-8")
    return [p.strip() for p in text.split("\n\n")[skip:] if p.strip()]
