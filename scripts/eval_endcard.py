# -*- coding: utf-8 -*-
"""向落点写：给定结尾一句，看模型能否自然落到那句上。"""
import argparse, json, os, re, statistics, sys, urllib.request
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flavor import analyze, normalize
from paths import QUALITY

SYS = ("你是中文小说家。先在心里倒推：这一段要走到哪一句，再从哪一件小事起笔。"
       "只写动作、物件、光与声音，不解释心理，不总结。风格：俄国旧译。只输出正文。")

ENDS = [
 "他把灯罩旋紧，听见玻璃和铜碰了一下。",
 "雪停了，屋檐上的水一滴一滴落下来。",
 "她把手帕叠好，放回口袋，没有说话。",
 "桌上的茶凉了，他没有喝。",
]


def maxrep(s, n=8):
    s = re.sub(r"\s+", "", s)
    c = Counter(s[i:i + n] for i in range(len(s) - n + 1))
    return max(c.values()) if c else 0


def call(base, model, user, temp=0.7, rp=1.2):
    body = {"model": model, "messages": [{"role": "system", "content": SYS},
                                         {"role": "user", "content": user}],
            "max_tokens": 460, "temperature": temp, "top_p": 0.9, "repeat_penalty": rp}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        j = json.loads(r.read().decode())
    return (j["choices"][0]["message"].get("content") or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8089")
    ap.add_argument("--model", default="fluorestsentsiya-3b-preview")
    ap.add_argument("--tag", default="endcard")
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()
    rows = []
    for end in ENDS:
        user = f"【落点】这一段要落在这句话上：\n{end}\n\n写一段小说，从一件具体的小事起笔，最后自然落到这句话，不要生硬拼接。"
        for k in range(args.n):
            try:
                txt = normalize(call(args.base, args.model, user))
            except Exception as e:
                print("ERR", e)
                txt = ""
            hit = txt.rstrip().endswith(end) or end.rstrip("。") in txt
            m = analyze(txt) if txt else None
            rep = maxrep(txt)
            rows.append({"end": end, "k": k, "out": txt, "hit": hit, "rep": rep, "m": m})
            print(f"[{args.tag}] {end[:10]} #{k} 字={len(txt)} 落到={hit} rep8={rep} "
                  f"ai={m['ai_score'] if m else '-'}", flush=True)
    if rows:
        print(f"\n{args.tag}: 落点命中 {sum(r['hit'] for r in rows)}/{len(rows)}  "
              f"复读率={sum(1 for r in rows if r['rep']>=3)/len(rows):.0%}  "
              f"平均字数={statistics.mean(len(r['out']) for r in rows):.0f}  "
              f"ai均值={statistics.mean(r['m']['ai_score'] for r in rows if r['m']):.1f}")
    with open(os.path.join(str(QUALITY), f"endcard_{args.tag}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
