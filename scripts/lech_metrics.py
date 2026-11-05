# -*- coding: utf-8 -*-
"""Lech Mazur benchmark metrics: format compliance, prose stats, elements, pairwise.

Upstream method notes
---------------------
* Stories must be 600-800 words, plain-ASCII English, wrapped in <story></story>,
  with a cumulative [word count] after every sentence.
* Comparisons are shown in both orders; choices are combined into a
  position-corrected relative rating.
Here, with two writer models and one evaluator, we report:
  - order-consistent win/loss/tie after combining AB and BA
  - position-bias rate
  - a two-model probit (Thurstone-style) relative score
"""
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from common import clean_story, read_jsonl, word_count, write_json  # noqa: E402

MIN_W, MAX_W = 600, 800


def sentence_stats(text):
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    lens = [word_count(s) for s in sents]
    return {
        "sent_n": len(sents),
        "sent_mean": round(statistics.mean(lens), 1) if lens else 0,
        "sent_cv": round(statistics.pstdev(lens) / statistics.mean(lens), 2) if lens and statistics.mean(lens) else 0,
    }


def rep4(text):
    toks = re.findall(r"[A-Za-z']+", text.lower())
    if len(toks) < 8:
        return 0.0
    grams = [" ".join(toks[i:i + 4]) for i in range(len(toks) - 3)]
    from collections import Counter
    c = Counter(grams)
    return round(sum(v - 1 for v in c.values() if v > 1) / len(grams), 3)


def ttr(text):
    toks = re.findall(r"[A-Za-z']+", text.lower())
    return round(len(set(toks)) / len(toks), 3) if toks else 0.0


def analyze_story(raw):
    body = clean_story(raw)
    words = word_count(body)
    counters = re.findall(r"\[\s*(\d+)\s*\]", raw or "")
    ascii_chars = sum(1 for ch in (raw or "") if ord(ch) < 128)
    total_chars = max(1, len(raw or ""))
    has_tag = bool(re.search(r"<story>", raw or "", re.I)) and bool(re.search(r"</story>", raw or "", re.I))
    return {
        "raw_chars": len(raw or ""),
        "words": words,
        "in_word_range": MIN_W <= words <= MAX_W,
        "has_story_tags": has_tag,
        "ascii_ratio": round(ascii_chars / total_chars, 3),
        "plain_ascii": ascii_chars == total_chars,
        "counter_markers": len(counters),
        "last_counter": int(counters[-1]) if counters else None,
        "counter_matches_words": (abs(int(counters[-1]) - words) <= 15) if counters else False,
        "rep4": rep4(body),
        "ttr": ttr(body),
        "english_ratio": round(len(re.findall(r"[A-Za-z]", body)) / max(1, len(body)), 3),
        **sentence_stats(body),
    }


def main():
    prompts = {p["id"]: p for p in read_jsonl(os.path.join(ROOT, "bench", "lech_prompts.jsonl"))}
    tags = sys.argv[1:] or ["filament-3b", "qwen25-3b-base"]
    stories = {}
    for tag in tags:
        stories[tag] = {r["id"]: r for r in read_jsonl(os.path.join(ROOT, "stories", "lech_%s.jsonl" % tag))}

    result = {"prompts": len(prompts), "models": {}, "pairwise": {}}
    for tag in tags:
        rows = []
        for pid, rec in sorted(stories[tag].items()):
            m = analyze_story(rec["story_raw"])
            m["prompt_id"] = pid
            m["generation_seconds"] = rec.get("elapsed_s")
            rows.append(m)
        if not rows:
            continue
        agg = {}
        for k in ("words", "ascii_ratio", "counter_markers", "rep4", "ttr", "sent_mean", "sent_cv", "english_ratio"):
            agg[k] = round(statistics.mean(r[k] for r in rows), 3)
        agg["n"] = len(rows)
        agg["in_word_range_pct"] = round(100 * sum(r["in_word_range"] for r in rows) / len(rows), 1)
        agg["has_story_tags_pct"] = round(100 * sum(r["has_story_tags"] for r in rows) / len(rows), 1)
        agg["plain_ascii_pct"] = round(100 * sum(r["plain_ascii"] for r in rows) / len(rows), 1)
        agg["counter_match_pct"] = round(100 * sum(r["counter_matches_words"] for r in rows) / len(rows), 1)
        result["models"][tag] = {"aggregate": agg, "per_story": rows}

    # elements judged by the critic
    el = read_jsonl(os.path.join(ROOT, "scores", "lech_elements.jsonl"))
    for tag in tags:
        vals = [r["count"] for r in el if r["model"] == tag and isinstance(r["count"], int)]
        ints = [r["integration_score"] for r in el if r["model"] == tag and isinstance(r["integration_score"], (int, float))]
        if vals:
            result["models"].setdefault(tag, {})["elements_used_mean"] = round(statistics.mean(vals), 2)
            result["models"][tag]["elements_n"] = len(vals)
        if ints:
            result["models"][tag]["integration_score_mean"] = round(statistics.mean(ints), 2)

    # pairwise, combine both orders
    pairs = read_jsonl(os.path.join(ROOT, "scores", "lech_pairwise.jsonl"))
    by_prompt = {}
    for r in pairs:
        by_prompt.setdefault(r["prompt_id"], []).append(r)
    wins = {tags[0]: 0, tags[1]: 0}
    ties = 0
    pos_first = 0
    total_j = 0
    detail = []
    for pid, recs in sorted(by_prompt.items()):
        recs = [r for r in recs if r["winner_model"]]
        if not recs:
            continue
        models_won = [r["winner_model"] for r in recs]
        total_j += len(recs)
        pos_first += sum(1 for r in recs if (r["order"] == "AB" and r["winner_label"] == "A")
                         or (r["order"] == "BA" and r["winner_label"] == "A"))
        if len(set(models_won)) == 1 and len(models_won) >= 2:
            wins[models_won[0]] += 1
            outcome = models_won[0]
        elif len(recs) == 1:
            wins[models_won[0]] += 1
            outcome = models_won[0]
        else:
            ties += 1
            outcome = "tie"
        detail.append({"prompt_id": pid, "outcome": outcome,
                       "orders": {r["order"]: r["winner_model"] for r in recs}})
    n = wins[tags[0]] + wins[tags[1]] + ties
    win_rate = (wins[tags[0]] + 0.5 * ties) / n if n else 0.0
    z = statistics.NormalDist().inv_cdf(min(0.999, max(0.001, win_rate)))
    result["pairwise"] = {
        "n_prompts": n,
        "wins": wins, "ties": ties,
        "win_rate_first_vs_second": round(win_rate, 3),
        "position_first_rate": round(pos_first / total_j, 3) if total_j else None,
        "probit_relative_score_first": round(z, 3),
        "detail": detail,
        "note": "probit relative score is the two-model Thurstone analogue; upstream reports a "
                "multi-model zero-centred scale, so only the sign/gap is comparable.",
    }
    write_json(os.path.join(ROOT, "reports", "lech_metrics.json"), result)
    print(json.dumps({"models": {k: v.get("aggregate", {}) for k, v in result["models"].items()},
                      "pairwise": {k: v for k, v in result["pairwise"].items() if k != "detail"}},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
