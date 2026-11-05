# -*- coding: utf-8 -*-
"""POLLUX 式风格评测（混合协议）：客观项用规则，主观项用评委。

10 条标准里，4 条可规则化（句长参差、破折号、现代词、网络用语），其余 6 条交本地 critic。
规则项可复现、零随机；主观项保留 LLM 评判。参考 POLLUX 的 criteria-based 评测思路。
"""
import argparse, json, os, re, statistics, sys, urllib.request
from paths import QUALITY

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODERN = ["手机", "电脑", "地铁", "咖啡", "公司", "项目", "数据", "网络", "屏幕", "电梯", "空调",
          "汽车", "火车", "电影", "电视", "电话", "银行", "超市", "医院", "学校", "办公室", "打印机"]
NET = re.compile(r"[A-Za-z]{3,}|[\U0001F300-\U0001FAFF]|[\u2600-\u27BF]|yyds|绝绝子|破防|栓Q|emo|U1S1")

SUBJ = [
    "以动作和物件推进：每段至少有一个具体动作或具体物件，不靠心理形容词推进",
    "对话简短，提示语简单（如“他说”“她说”），不用“他激动地说道”这类副词堆叠",
    "出现具体的时代物件（灯、茶炊、马车、制服、卢布、雪橇、圣像等至少一种）",
    "场景转换有过渡（时间、地点或动作承接），不是硬切",
    "结尾落在具体动作或物件上，不总结、不抒情议论",
    "没有模板腔套话（如“在这个……的时代”“不禁”“仿佛在诉说着”“令人不禁”）",
]
JUDGE_SYS = "你是中文文学编辑。按给定标准逐条判断文本是否满足，只输出 JSON。"


def call(base, model, system, user, temp=0.3, n=900):
    body = {"model": model, "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": user}],
            "max_tokens": n, "temperature": temp, "top_p": 0.9}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        j = json.loads(r.read().decode())
    return (j["choices"][0]["message"].get("content") or "").strip()


def rule_checks(t):
    sents = [s for s in re.split(r"(?<=[。！？])", t) if s.strip()]
    lens = [len(s) for s in sents] or [0]
    c1 = (max(lens) >= 30) and (min(lens) <= 14)
    c3 = ("—" not in t) and ("……" not in t)
    c6 = not any(w in t for w in MODERN)
    c7 = not NET.search(t)
    return [c1, c3, c6, c7]


def judge(critic, text):
    crit = "\n".join(f"{i+1}. {c}" for i, c in enumerate(SUBJ))
    prompt = (f"【标准】\n{crit}\n\n【文本】\n{text[:4000]}\n\n"
              f"逐条判断，共 6 项。只输出 JSON：{{\"pass\": [true,false,... 共 6 项]}}")
    for temp in (0.1, 0.3):
        raw = call(critic, "critic", JUDGE_SYS, prompt, n=300, temp=temp)
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                p = json.loads(m.group(0)).get("pass") or []
                if len(p) == 6:
                    return [bool(x) for x in p]
            except Exception:
                pass
        nums = re.findall(r"[01]", raw)
        if len(nums) >= 6:
            return [bool(int(x)) for x in nums[:6]]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--critic", default="http://127.0.0.1:8092")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.prompts, encoding="utf-8") if l.strip()][: args.n]
    per, details = [], []
    for r in rows:
        txt = call(args.endpoint, args.model, r.get("system", ""), r["user"], temp=args.temp)
        if len(txt) < 80:
            continue
        rp = rule_checks(txt)
        jp = judge(args.critic, txt)
        if jp is None:
            continue
        row = [rp[0], jp[0], rp[1], jp[1], jp[2], rp[2], rp[3], jp[3], jp[4], jp[5]]
        per.append(row)
        details.append({"chars": len(txt), "score": sum(row), "pass": row})
    names = ["句长参差(规则)", "动作/物件推进", "无破折号省略号(规则)", "对话提示语简短",
             "时代物件", "无现代词(规则)", "无网络用语(规则)", "场景过渡", "结尾落在动作/物件", "无AI套话"]
    out = {"tag": args.tag, "n": len(per), "temp": args.temp,
           "style_score": round(statistics.mean(sum(x) for x in per), 2) if per else None,
           "by_criterion": {names[i]: round(sum(x[i] for x in per) / len(per), 2) for i in range(10)} if per else {},
           "details": details}
    p = os.path.join(str(QUALITY), f"pollux2_{args.tag}.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ("tag", "n", "style_score", "by_criterion")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
