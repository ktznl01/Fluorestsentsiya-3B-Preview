# -*- coding: utf-8 -*-
"""中文“模板腔”量化：符号密度、套话密度、成语密度、句长参差、句首重复、说破心理。

参考（GitHub）：
  把模板腔量化：词表密度 / 句式重复率 / 模板腔密度 / 结构指纹
  hongcha1101/de-aigc-ch          中文降 AIGC 率的 skill（主矛盾优先于词表句式）
  zhuyansen/awesome-humanizer-skills  人味/反 slop 清单

用法：
  python ai_flavor.py --file 正文.md
  python ai_flavor.py --jsonl bench/quality/style3b.jsonl --field out
"""
import argparse, json, math, re, statistics, sys

CLICHE = ["仿佛", "似乎", "一丝", "微微", "不禁", "缓缓", "静静地", "深深地", "悄悄",
          "瞬间", "刹那间", "空气仿佛", "凝固", "心中", "心里", "眼中", "目光", "嘴角",
          "泛起", "涟漪", "五味杂陈", "难以言喻", "无法言说", "若有所思", "意味深长",
          "不容置疑", "油然而生", "涌上心头", "如释重负", "恍然", "顿悟", "沉默良久"]
IDIOM = ["栩栩如生", "历历在目", "络绎不绝", "川流不息", "美不胜收", "心旷神怡", "波澜壮阔",
         "跌宕起伏", "扣人心弦", "引人入胜", "身临其境", "惟妙惟肖", "淋漓尽致", "恰到好处",
         "无微不至", "井然有序", "一丝不苟", "全神贯注", "聚精会神", "不由自主", "情不自禁",
         "若有所思", "恍然大悟", "不知所措", "莫名其妙", "无可奈何", "无能为力", "小心翼翼",
         "依依不舍", "念念不忘", "历久弥新", "源远流长", "博大精深", "耐人寻味", "回味无穷"]
PSYCHO = ["他感到", "她感到", "他觉得自己", "她觉得自己", "他意识到", "她意识到",
          "他明白", "她明白", "他知道自己", "她知道她", "内心", "心底", "思绪"]
CONNECT = ["然而", "于是", "接着", "随后", "与此同时", "总而言之", "总之", "不难看出",
           "由此可见", "值得注意的是", "毫无疑问"]


def _per_k(count, n):
    return round(count / max(1, n) * 1000, 2)


def normalize(text):
    """把模板腔的破折号/省略号收掉，保留句读；行尾的逗号改成句号。"""
    t = text or ""
    t = t.replace("——", "，").replace("……", "，").replace("…", "，")
    t = re.sub(r"[，]{2,}", "，", t)
    t = re.sub(r"，(?=[。！？])", "", t)
    t = re.sub(r"。，", "。", t)
    lines = []
    for ln in t.splitlines():
        s = ln.rstrip()
        if s.endswith("，") or s.endswith("、") or s.endswith("；"):
            s = s[:-1] + "。"
        lines.append(s)
    t = "\n".join(lines)
    t = re.sub(r"[ \t]+", "", t)
    return t.strip()


def analyze(text):
    t = re.sub(r"\s+", "", text or "")
    n = len(t)
    if n < 20:
        return None
    sents = [s for s in re.split(r"(?<=[。！？…])", t) if s.strip()]
    lens = [len(s) for s in sents] or [n]
    mean = statistics.mean(lens)
    cv = (statistics.pstdev(lens) / mean) if mean else 0

    sym = {
        "dash": t.count("——") + t.count("—"),
        "ellipsis": t.count("……") + t.count("…"),
        "excl": t.count("！"),
        "quest": t.count("？"),
        "semi": t.count("；"),
        "quote": t.count("“"),
    }
    starts = {}
    for s in sents:
        k = s[:2]
        starts[k] = starts.get(k, 0) + 1
    start_repeat = max(starts.values()) / len(sents) if sents else 0

    cliche = sum(t.count(w) for w in CLICHE)
    idiom = sum(t.count(w) for w in IDIOM)
    psycho = sum(t.count(w) for w in PSYCHO)
    conn = sum(t.count(w) for w in CONNECT)
    dialogue = sum(t.count(w) for w in ("说", "问", "道", "答"))
    punct = sym["dash"] + sym["ellipsis"]

    # 粗略排版对称：三个以上相邻句子长度几乎相等
    tri = 0
    for i in range(len(lens) - 2):
        a, b, c = lens[i:i + 3]
        if max(a, b, c) - min(a, b, c) <= 2 and a >= 6:
            tri += 1

    # 注意：本项目的文体本来就用破折号/省略号（契诃夫中译即如此），
    # 所以标点只占很小权重；判别力在句长参差、套话、说破心理、句首重复。
    score = (
        _per_k(cliche, n) * 1.4
        + _per_k(idiom, n) * 1.2
        + _per_k(psycho, n) * 1.8
        + _per_k(conn, n) * 1.0
        + min(40, _per_k(punct, n)) * 0.7
        + max(0, 0.72 - cv) * 45
        + max(0, start_repeat - 0.22) * 45
        + tri * 2.0
        + _per_k(sym["semi"] + sym["excl"], n) * 0.15
    )
    return {
        "chars": n, "sent_mean": round(mean, 1), "sent_cv": round(cv, 2),
        "dash_per_k": _per_k(sym["dash"], n), "ellipsis_per_k": _per_k(sym["ellipsis"], n),
        "punct_per_k": _per_k(punct, n),
        "excl_per_k": _per_k(sym["excl"], n), "semi_per_k": _per_k(sym["semi"], n),
        "cliche_per_k": _per_k(cliche, n), "idiom_per_k": _per_k(idiom, n),
        "psycho_per_k": _per_k(psycho, n), "connect_per_k": _per_k(conn, n),
        "start_repeat": round(start_repeat, 2), "tri_sym": tri,
        "ai_score": round(score, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--jsonl")
    ap.add_argument("--field", default="out")
    args = ap.parse_args()
    texts = []
    if args.file:
        raw = open(args.file, encoding="utf-8").read()
        paras = [p.strip() for p in raw.split("\n\n") if p.strip() and not p.startswith("#")]
        texts = ["\n".join(paras[i:i + 3]) for i in range(0, len(paras), 3)]
    if args.jsonl:
        for line in open(args.jsonl, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get(args.field):
                    texts.append(r[args.field])
    rows = [analyze(t) for t in texts]
    rows = [r for r in rows if r]
    if not rows:
        sys.exit("no text")
    keys = ["ai_score", "chars", "sent_mean", "sent_cv", "dash_per_k", "ellipsis_per_k",
            "excl_per_k", "cliche_per_k", "idiom_per_k", "psycho_per_k", "start_repeat"]
    print(" | ".join(keys))
    print(" | ".join(str(round(statistics.mean(r[k] for r in rows), 2)) for k in keys))
    for k in ("ai_score", "dash_per_k", "ellipsis_per_k", "cliche_per_k", "idiom_per_k"):
        vals = sorted(r[k] for r in rows)
        print(f"  {k}: min={vals[0]} p50={vals[len(vals)//2]} max={vals[-1]}")


if __name__ == "__main__":
    main()
