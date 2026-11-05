# -*- coding: utf-8 -*-
"""首尾呼应 / 细节复用 评测：给定物件，看末句是否回到它、正文是否真的用了它。"""
import argparse, json, os, re, statistics, sys, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flavor import normalize
from paths import QUALITY

SYS = ("你是中文小说家。只写动作、物件、光与声音，不解释。风格：俄国旧译。只输出正文。")
VERBS = "拿放给开关打扔烧撕戴递摸掏塞收捡拾握捏用"

ECHO_CASES = ["灯", "手帕", "信", "戒指", "雪", "钥匙", "照片", "镜子"]
DETAIL_CASES = ["雨伞", "茶", "箱子", "眼镜", "刀", "信纸", "帽子", "钟"]


def call(base, model, prompt, temp=0.7, rp=1.2):
    body = {"model": model, "messages": [{"role": "system", "content": SYS},
                                         {"role": "user", "content": prompt}],
            "max_tokens": 460, "temperature": temp, "top_p": 0.9, "repeat_penalty": rp}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        j = json.loads(r.read().decode())
    return (j["choices"][0]["message"].get("content") or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8093")
    ap.add_argument("--model", default="fluorestsentsiya-3b-preview")
    ap.add_argument("--tag", default="echo")
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()
    rows = []
    for obj in ECHO_CASES:
        for k in range(args.n):
            p = f"写一段小说：开头让“{obj}”出现，最后一句要回到“{obj}”，但不要生硬重复原句。约二百字。"
            try:
                t = normalize(call(args.base, args.model, p))
            except Exception as e:
                print("ERR", e); t = ""
            ss = [x for x in re.split(r"(?<=[。！？])", t) if x.strip()]
            last = ss[-1] if ss else ""
            tail_hit = obj in last
            uses = t.count(obj)
            rows.append({"kind": "echo", "obj": obj, "out": t, "hit": tail_hit, "uses": uses})
            print(f"[echo] {obj} 末句含={tail_hit} 出现{uses}次 字={len(t)}", flush=True)
    for obj in DETAIL_CASES:
        for k in range(args.n):
            p = f"写一段小说：开头让“{obj}”出现，后面它必须真的被用到、造成一个后果。约二百字。"
            try:
                t = normalize(call(args.base, args.model, p))
            except Exception:
                t = ""
            uses = t.count(obj)
            second = t.find(obj, t.find(obj) + 1)
            win = t[max(0, second - 10): second + 10] if second > 0 else ""
            used = second > 0 and any(v in win for v in VERBS)
            rows.append({"kind": "detail", "obj": obj, "out": t, "uses": uses, "used": used})
            print(f"[detail] {obj} 出现{uses}次 带动作={used} 字={len(t)}", flush=True)
    e = [r for r in rows if r["kind"] == "echo"]
    d = [r for r in rows if r["kind"] == "detail"]
    if e:
        print(f"\n{args.tag} echo: 末句命中 {sum(r['hit'] for r in e)}/{len(e)} "
              f"平均出现 {statistics.mean(r['uses'] for r in e):.1f} 次")
    if d:
        print(f"{args.tag} detail: 二次出现且带动作 {sum(r['used'] for r in d)}/{len(d)} "
              f"平均出现 {statistics.mean(r['uses'] for r in d):.1f} 次")
    outdir = str(QUALITY)
    with open(os.path.join(outdir, f"echo_{args.tag}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
