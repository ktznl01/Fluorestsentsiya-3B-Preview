# -*- coding: utf-8 -*-
"""长文生成管线：列拍 → 逐拍写 → 拼装 → 连贯性度量。

用法：
  python longform.py --premise "..." --target 3000 --out projects\\长夜\\正文.md
"""
import argparse, json, os, re, statistics, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flavor import normalize
from story_metrics import analyze as story
from scene_jump import analyze as jump

SYS = ("你是中文小说家，译笔取法旧译：句子沉实，长短参差，白描，只写看得见的动作与物，"
       "不议论，不总结。风格：俄国旧译。只输出正文。")
PLAN_SYS = ("你是小说编辑。读现有正文（可能为空），列出接下来要写的五拍，每拍一行，"
            "以“这一拍：”开头，每行不超过二十字，写具体动作，不要解释、不要编号。")


def call(base, model, sys_p, user, temp=0.7, n=600, rp=1.2):
    body = {"model": model, "messages": [{"role": "system", "content": sys_p},
                                         {"role": "user", "content": user}],
            "max_tokens": n, "temperature": temp, "top_p": 0.9, "repeat_penalty": rp}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        j = json.loads(r.read().decode())
    return normalize((j["choices"][0]["message"].get("content") or "").strip())


def dedup_text(t):
    """句级 + 段级去重。"""
    parts = [x.strip() for x in re.split(r"\n\s*\n", t) if x.strip()]
    seen_p, paras = set(), []
    for p in parts:
        k = re.sub(r"\s+", "", p)[:24]
        if len(k) >= 12 and k in seen_p:
            continue
        if len(k) >= 12:
            seen_p.add(k)
        paras.append(p)
    out, seen_s = [], set()
    for p in paras:
        sents = re.split(r"(?<=[。！？])", p)
        keep = []
        for s0 in sents:
            k = re.sub(r"\s+", "", s0)
            if len(k) >= 12 and k in seen_s:
                continue
            if len(k) >= 12:
                seen_s.add(k)
            keep.append(s0)
        out.append("".join(keep).strip())
    return "\n\n".join(x for x in out if x)


def longest_shared(a, b, n=14):
    """a 与 b 最长公共 n-gram 命中的比例（近似）。"""
    ga = set(a[i:i + n] for i in range(max(0, len(a) - n + 1)))
    gb = set(b[i:i + n] for i in range(max(0, len(b) - n + 1)))
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / max(1, len(gb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8087")
    ap.add_argument("--model", default="fluorestsentsiya-3b-preview")
    ap.add_argument("--premise", required=True)
    ap.add_argument("--target", type=int, default=3000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-beats", type=int, default=16)
    args = ap.parse_args()

    text = ""
    beats_total = 0
    t0 = time.time()
    while len(re.sub(r"\s+", "", text)) < args.target and beats_total < args.max_beats:
        head = text[-1800:] if text else ""
        plan = call(args.base, args.model, PLAN_SYS,
                    f"【已有正文】\n{head}\n\n【开头设想】{args.premise}\n\n只输出五行。", temp=0.5, n=200)
        beats = [x.strip() for x in plan.splitlines() if x.strip().startswith("这一拍")][:5]
        if not beats:
            break
        for b in beats:
            if len(re.sub(r"\s+", "", text)) >= args.target:
                break
            beat_txt = re.sub(r"^这一拍[：:]\s*", "", b).strip()
            uw = (f"【开头设想】{args.premise}\n\n【已有正文】\n{text[-2200:]}\n\n"
                  f"只写接下来这一件事：{beat_txt}。约二百五十字。\n"
                  f"从正文最后一句之后写起，不重复上文，不写别的事。")
            piece = call(args.base, args.model, SYS, uw, temp=0.7, n=520)
            piece = re.sub(r"^这一拍[：:].*$", "", piece, flags=re.M).strip()
            # 与已有正文重合度过高就换一稿
            if piece and text and longest_shared(text, piece) > 0.25:
                uw2 = uw + "\n上一稿重复了上文，请换一个写法：写新的动作，不要复述已经写过的句子。"
                p2 = call(args.base, args.model, SYS, uw2, temp=0.9, n=520)
                p2 = re.sub(r"^这一拍[：:].*$", "", p2, flags=re.M).strip()
                if p2 and longest_shared(text, p2) <= 0.25:
                    piece = p2
            if piece:
                text = dedup_text((text + "\n\n" + piece).strip())
                beats_total += 1
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write(text)
    m = story(text) or {}
    j = jump(text) or {}
    print(json.dumps({"chars": len(re.sub(r"\s+", "", text)), "paras": text.count("\n\n") + 1,
                      "beats": beats_total, "sec": round(time.time() - t0, 1),
                      "sent_mean": m.get("sent_mean"), "rep4": m.get("rep4"),
                      "cohesion": j.get("cohesion"), "jump_rate": j.get("jump_rate")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
