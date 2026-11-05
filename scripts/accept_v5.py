# -*- coding: utf-8 -*-
"""v4 验收：v3 全部功能 + 场景承接（cohesion）与跳跃率。"""
import argparse, json, os, re, statistics, sys, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flavor import normalize
from scene_jump import analyze as jump_analyze
from paths import QUALITY

RU = ("你是中文小说家，译笔取法旧译：句子沉实，长短参差，白描，只写看得见的动作与物。"
      "不用破折号和省略号。风格：俄国旧译。只输出正文。")
LEDGER = ("【账本】\n人物：伊利亚（验灯工）；马特维（点灯人，右手只剩三根手指）\n"
          "规则：不得新增账本外的人名；时间只能前进。")
PLAN_SYS = ("你是小说编辑。读现有正文，列出接下来要写的三拍，每拍一行，以“这一拍：”开头，"
            "每行不超过二十字，写具体动作，不要解释、不要编号。")
TAIL = "灯还亮着。窗外的天黑透了。窗台上那一排修好的灯，一只也没有开。"

PROMPTS = ["夜里十一点，验灯间，二百只灯泡同时亮着。他坐在长桌尽头验灯。写三百字。",
           "夜里两点，他下班回家，街上积雪，路灯隔五十步一盏。写三百字。",
           "工头推门进来，把报表摊在桌上。写三百字。",
           "一只灯泡在他手里炸了。写三百字。",
           "他发烧躺在床上，数屋顶的灯丝。写三百字。",
           "他提着工具箱走出厂门，白天的灯还亮着。写三百字。"]


def call(base, model, sys_p, user, temp=0.7, n=700):
    body = {"model": model, "messages": [{"role": "system", "content": sys_p},
                                         {"role": "user", "content": user}],
            "max_tokens": n, "temperature": temp, "top_p": 0.9, "repeat_penalty": 1.2}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        j = json.loads(r.read().decode())
    return normalize((j["choices"][0]["message"].get("content") or "").strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8093")
    ap.add_argument("--model", default="fluorestsentsiya-3b-preview")
    ap.add_argument("--tag", default="v5")
    args = ap.parse_args()
    res = {}

    t = call(args.base, args.model, "你是中文小说家。遵守账本。只输出正文。",
             LEDGER + "\n\n【已有正文】\n" + TAIL + "\n\n接着写，让马特维来说一句话。", n=320)
    res["ledger"] = ("手指" in t)

    p = call(args.base, args.model, PLAN_SYS,
             "【已有正文】\n" + TAIL + "\n\n【要求】接着写下去。\n\n只输出三行。", temp=0.5, n=180)
    res["plan_beats"] = len([x for x in p.splitlines() if x.strip().startswith("这一拍")])

    ends = ["他把灯罩旋紧，听见玻璃和铜碰了一下。", "雪停了，屋檐上的水一滴一滴落下来。",
            "她把手帕叠好，放回口袋，没有说话。", "桌上的茶凉了，他没有喝。"]
    hits = 0
    for e in ends:
        o = call(args.base, args.model, RU,
                 f"【落点】这一段要落在这句话上：\n{e}\n\n写一段小说，最后自然落到这句话。", n=440)
        if e.rstrip("。") in o:
            hits += 1
    res["landing"] = f"{hits}/{len(ends)}"

    long_t = call(args.base, args.model, RU,
                  "写一段小说，五百字左右：夜里两点，验灯工下班回家，街上积雪，路灯隔五十步一盏，他数着走。", n=900)
    res["long_chars"] = len(long_t)

    outs = []
    for pr in PROMPTS:
        o = call(args.base, args.model, RU, pr, n=700)
        outs.append(o)
    ms = [jump_analyze(o) for o in outs]
    ms = [m for m in ms if m]
    res["cohesion"] = round(statistics.mean(m["cohesion"] for m in ms), 4)
    res["jump_rate"] = round(statistics.mean(m["jump_rate"] for m in ms), 3)

    print(json.dumps(res, ensure_ascii=False, indent=1))
    ok = (res["ledger"] and res["plan_beats"] >= 2 and hits >= 1 and res["long_chars"] >= 300
          and res["cohesion"] >= 0.006 and res["jump_rate"] <= 0.07)
    print("PASS" if ok else "FAIL")
    with open(os.path.join(str(QUALITY), f"accept_{args.tag}.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
