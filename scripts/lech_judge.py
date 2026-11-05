# -*- coding: utf-8 -*-
"""Lech Mazur benchmark judging, local adaptation.

Upstream: matched story pairs are read by evaluator models in BOTH orders, and
their choices are combined into a position-corrected relative rating (Thurstone)
over many writer models. Locally we have one evaluator (the WritingBench critic
Qwen-7B), so we run both presentation orders and record:
  * winner per order (A / B / tie)
  * order-consistency (a disagreement = position-bias-driven tie)
  * element coverage per story (10 mandatory elements)

Outputs:
  scores/lech_pairwise_<tag>.jsonl   one record per (prompt, order)
  scores/lech_elements_<tag>.jsonl   one record per story
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from common import (LlamaServer, append_jsonl, clean_story, parse_json_object,  # noqa: E402
                    read_jsonl, word_count)

ELEMENTS = [
    "Character", "Object", "Core Concept", "Attribute", "Action",
    "Method", "Setting", "Timeframe", "Motivation", "Tone",
]

JUDGE_SYSTEM = ("You are an expert creative-writing judge. You compare two stories written to "
                "the same constrained brief and decide which one is better, strictly following "
                "the brief's grading factors. You output only JSON.")

JUDGE_PROMPT = """Two writers were given the same brief. Each brief lists ten mandatory narrative elements
(character, object, core concept, attribute, action, method, setting, timeframe, motivation, tone)
and asks for a 600-800 word story that integrates all of them naturally.

Grading factors: Story Integration, Character Development & Motivation, Plot Structure & Coherence,
World & Atmosphere, Storytelling Impact & Craft, Authenticity & Originality.

Judge which story is better. Reward integration of the required elements into plot, character and
theme over keyword mention; penalise stories that ignore the word range, break the output format
(<story></story> tags, plain-ASCII English, cumulative word counts), drift off-brief, or read as
repetitive filler.

--- BRIEF ---
{brief}

--- STORY {first} ---
{story_first}

--- STORY {second} ---
{story_second}

Return only JSON:
{{"winner": "{first}" or "{second}" or "tie", "reason": "two or three concrete sentences referencing both stories"}}
"""

ELEMENT_PROMPT = """You check whether a story actually integrates the ten mandatory narrative elements of its brief.
Be strict: an element counts as integrated only when a concrete sentence makes it drive an action, an image, a choice
or the theme. A bare keyword mention, a label, or a title does not count; if the story is short and only gestures at
an element, it does not count.

List ONLY the element names that are missing or merely decorative, using exactly these ten names:
Character, Object, Core Concept, Attribute, Action, Method, Setting, Timeframe, Motivation, Tone.

--- BRIEF ---
{brief}

--- STORY ---
{story}

Return only JSON, with the missing list built from what you actually find in THIS story:
{{"missing": ["<element name>", ...], "integration_score": <1-10>, "notes": "one sentence naming the strongest and weakest element"}}
"""


def load_stories(model_tag):
    path = os.path.join(ROOT, "stories", "lech_%s.jsonl" % model_tag)
    rows = read_jsonl(path)
    return {r["id"]: r for r in rows}


def judge_pair(srv, brief, story_first, story_second, retries=3):
    prompt = JUDGE_PROMPT.format(brief=brief, first="A", second="B",
                                 story_first=story_first, story_second=story_second)
    last = None
    for attempt in range(1, retries + 1):
        try:
            res = srv.chat([{"role": "system", "content": JUDGE_SYSTEM},
                            {"role": "user", "content": prompt}],
                           temperature=0.3, top_p=0.9, max_tokens=500)
            obj = parse_json_object(res["text"])
            if not obj:
                last = "no-json: " + res["text"][:160]
                continue
            w = str(obj.get("winner", "")).strip().upper()
            if w not in ("A", "B", "TIE"):
                last = "bad-winner: " + w
                continue
            return {"winner_label": w, "reason": str(obj.get("reason", ""))[:1200]}
        except Exception as e:  # noqa: BLE001
            last = str(e)
            time.sleep(2 * attempt)
    return {"winner_label": None, "reason": "FAILED: %s" % last}


def score_elements(srv, brief, story, retries=3):
    prompt = ELEMENT_PROMPT.format(brief=brief, story=story)
    last = None
    for attempt in range(1, retries + 1):
        try:
            res = srv.chat([{"role": "system", "content": JUDGE_SYSTEM},
                            {"role": "user", "content": prompt}],
                           temperature=0.2, top_p=0.9, max_tokens=500)
            obj = parse_json_object(res["text"])
            if not obj or "missing" not in obj:
                last = "no-json: " + res["text"][:160]
                continue
            missing = obj["missing"] if isinstance(obj["missing"], list) else []
            miss = set()
            for m in missing:
                for e in ELEMENTS:
                    if e.lower() in str(m).lower():
                        miss.add(e)
            used = {k: (k not in miss) for k in ELEMENTS}
            return {"used": used, "count": sum(used.values()),
                    "missing": sorted(miss),
                    "integration_score": obj.get("integration_score"),
                    "notes": str(obj.get("notes", ""))[:600]}
        except Exception as e:  # noqa: BLE001
            last = str(e)
            time.sleep(2 * attempt)
    return {"used": {k: None for k in ELEMENTS}, "count": None,
            "integration_score": None, "notes": "FAILED: %s" % last}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="two model tags")
    ap.add_argument("--prompts", default=os.path.join(ROOT, "bench", "lech_prompts.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-elements", action="store_true")
    args = ap.parse_args()

    prompts = {p["id"]: p for p in read_jsonl(args.prompts)}
    if args.limit:
        prompts = dict(list(prompts.items())[:args.limit])
    a_tag, b_tag = args.models[0], args.models[1]
    stories = {a_tag: load_stories(a_tag), b_tag: load_stories(b_tag)}

    pair_out = os.path.join(ROOT, "scores", "lech_pairwise.jsonl")
    el_out = os.path.join(ROOT, "scores", "lech_elements.jsonl")
    done_pair = {(r["prompt_id"], r["order"]) for r in read_jsonl(pair_out)}
    done_el = {(r["prompt_id"], r["model"]) for r in read_jsonl(el_out)}

    srv = LlamaServer(args.endpoint, timeout=1200)
    if not srv.health():
        sys.exit("critic server not healthy: " + args.endpoint)

    for pid in sorted(prompts):
        brief = prompts[pid]["prompt"]
        if pid not in stories[a_tag] or pid not in stories[b_tag]:
            print("skip", pid, "(missing story)")
            continue
        a = clean_story(stories[a_tag][pid]["story_raw"])
        b = clean_story(stories[b_tag][pid]["story_raw"])
        for order, first_tag in (("AB", a_tag), ("BA", b_tag)):
            key = (pid, order)
            if key in done_pair:
                continue
            if order == "AB":
                s1, s2 = a, b
            else:
                s1, s2 = b, a
            res = judge_pair(srv, brief, s1, s2)
            if res["winner_label"] == "A":
                winner = first_tag
            elif res["winner_label"] == "B":
                winner = b_tag if first_tag == a_tag else a_tag
            else:
                winner = None
            rec = {
                "prompt_id": pid, "order": order,
                "first_model": first_tag,
                "second_model": b_tag if first_tag == a_tag else a_tag,
                "winner_label": res["winner_label"],
                "winner_model": winner,
                "reason": res["reason"],
                "a_tag": a_tag, "b_tag": b_tag,
            }
            append_jsonl(pair_out, rec)
            print("  pair %-6s %s -> %s" % (pid, order, rec["winner_model"] or "tie/fail"))
            sys.stdout.flush()

        if args.skip_elements:
            continue
        for tag, story in ((a_tag, a), (b_tag, b)):
            if (pid, tag) in done_el:
                continue
            el = score_elements(srv, brief, story)
            rec = {"prompt_id": pid, "model": tag, "elements": el["used"],
                   "count": el["count"], "missing": el.get("missing", []),
                   "integration_score": el["integration_score"],
                   "notes": el["notes"], "words": word_count(story), "chars": len(story)}
            append_jsonl(el_out, rec)
            print("  elements %-6s %-14s %s/10" % (pid, tag, el["count"]))
            sys.stdout.flush()

    print("done ->", pair_out, el_out)


if __name__ == "__main__":
    main()
