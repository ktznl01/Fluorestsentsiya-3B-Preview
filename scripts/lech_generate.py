# -*- coding: utf-8 -*-
"""Generate stories for the Lech Mazur Creative Story-Writing Benchmark prompts.

Upstream method: every writer model receives the same fixed brief (10 mandatory
narrative elements + format rules) and must return a 600-800 word story. We keep
the brief verbatim; only the sampling stack differs (local llama.cpp).

Usage:
  python lech_generate.py --endpoint http://127.0.0.1:8087 --tag fluorestsentsiya-3b-preview
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from common import LlamaServer, append_jsonl, read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompts", default=os.path.join(ROOT, "bench", "lech_prompts.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-tokens", type=int, default=1600)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = args.out or os.path.join(ROOT, "stories", "lech_%s.jsonl" % args.tag)
    done = {r["id"] for r in read_jsonl(out)}
    prompts = read_jsonl(args.prompts)
    if args.limit:
        prompts = prompts[:args.limit]
    todo = [p for p in prompts if p["id"] not in done]
    print("[%s] %d prompts, %d done, %d to go -> %s" % (args.tag, len(prompts), len(done), len(todo), out))
    srv = LlamaServer(args.endpoint)
    if not srv.health():
        sys.exit("server not healthy: " + args.endpoint)

    for k, p in enumerate(todo, 1):
        t0 = time.time()
        try:
            res = srv.chat([{"role": "user", "content": p["prompt"]}],
                           temperature=args.temperature, top_p=args.top_p,
                           max_tokens=args.max_tokens, seed=42)
            text, err = res["text"], None
        except Exception as e:  # noqa: BLE001
            text, res, err = "", {"usage": {}, "timings": {}}, str(e)
        rec = {
            "id": p["id"],
            "model": args.tag,
            "prompt_file": p.get("prompt_file"),
            "story_raw": text,
            "chars": len(text),
            "elapsed_s": round(time.time() - t0, 1),
            "usage": res.get("usage", {}),
            "error": err,
        }
        append_jsonl(out, rec)
        print("  [%d/%d] %-6s %5d chars %5.1fs%s" % (k, len(todo), p["id"], len(text),
                                                     rec["elapsed_s"], (" ERR " + err) if err else ""))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
