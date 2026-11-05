# -*- coding: utf-8 -*-
"""扩大样本的正面对比：落点 12 题、长文 8 题、场景 10 题。"""
import argparse, json, os, re, statistics, sys, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_flavor import normalize
from story_metrics import analyze as story
from scene_jump import analyze as jump
from paths import QUALITY

SYS = ("你是中文小说家，译笔取法旧译：句子沉实，长短参差，白描，只写看得见的动作与物，"
       "不议论，不总结。不用破折号和省略号。风格：俄国旧译。只输出正文。")

ENDS = ["他把灯罩旋紧，听见玻璃和铜碰了一下。", "雪停了，屋檐上的水一滴一滴落下来。",
        "她把手帕叠好，放回口袋，没有说话。", "桌上的茶凉了，他没有喝。",
        "他把门带上，站在台阶上听了一会儿。", "火车开走了，站台上只剩一盏灯。",
        "她把信折好，压在枕头底下。", "他数到第九盏灯，停住了。",
        "炉子灭了，屋里比外面还冷。", "他把钱数了两遍，还是少一张。",
        "她把手套摘下来，放在膝上。", "窗上的霜化了一半，露出街。"]

LONG = ["写一段小说，五百字左右：夜里两点，验灯工下班回家，街上积雪，路灯隔五十步一盏，他数着走。",
        "写一段小说，五百字左右：工头把报表摊在桌上，指着其中一行，验灯工看着那行数字。",
        "写一段小说，五百字左右：他发烧躺在床上，数屋顶的灯丝。",
        "写一段小说，五百字左右：妹妹从外省来信，三行，要钱。",
        "写一段小说，五百字左右：他提着一只修好的灯泡，去给街上的点灯人换上。",
        "写一段小说，五百字左右：厂里要裁人，名单贴在门口，他站在人群后面。",
        "写一段小说，五百字左右：妻子病了两年，他在药房门口排队。",
        "写一段小说，五百字左右：他走出厂门，白天的灯还亮着，没有人去关。"]

SCENE = ["夜里十一点，验灯间，二百只灯泡同时亮着。他坐在长桌尽头验灯。写三百字。",
         "夜里两点，他下班回家，街上积雪。写三百字。",
         "工头推门进来，把报表摊在桌上。写三百字。",
         "一只灯泡在他手里炸了。写三百字。",
         "他发烧躺在床上，数屋顶的灯丝。写三百字。",
         "他提着工具箱走出厂门。写三百字。",
         "点灯人马特维在修第九盏灯。写三百字。",
         "学徒科利亚蹲在屋角，手里捏着一块冷面包。写三百字。",
         "他陪妻子去医院，走廊很白。写三百字。",
         "他把最后一只灯放回箱子，天快亮了。写三百字。"]


def call(base, model, user, temp=0.7, n=800):
    body = {"model": model, "messages": [{"role": "system", "content": SYS},
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
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--temp", type=float, default=0.7)
    args = ap.parse_args()

    hits = 0
    for e in ENDS:
        o = call(args.base, args.model, f"【落点】这一段要落在这句话上：\n{e}\n\n写一段小说，最后自然落到这句话。", temp=args.temp, n=460)
        if e.rstrip("。") in o:
            hits += 1
    lens = [len(call(args.base, args.model, p, temp=args.temp, n=1000)) for p in LONG]
    outs = [call(args.base, args.model, p, temp=args.temp, n=700) for p in SCENE]
    ms = [story(o) for o in outs]
    ms = [m for m in ms if m]
    js = [jump(o) for o in outs if o]
    js = [x for x in js if x]
    res = {"tag": args.tag,
           "landing": f"{hits}/{len(ENDS)}",
           "long_mean": round(statistics.mean(lens)),
           "long_min": min(lens),
           "cohesion": round(statistics.mean(x["cohesion"] for x in js), 4),
           "jump_rate": round(statistics.mean(x["jump_rate"] for x in js), 3),
           "rep4": round(statistics.mean(m["rep4"] for m in ms), 3),
           "sent_mean": round(statistics.mean(m["sent_mean"] for m in ms), 1)}
    print(json.dumps(res, ensure_ascii=False))
    with open(os.path.join(str(QUALITY), f"h2h_{args.tag}.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
