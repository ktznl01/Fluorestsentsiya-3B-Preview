# -*- coding: utf-8 -*-
"""场景跳跃检测：实体分布 + 词汇衔接 + 换场标记（entity grid 的近似实现）。

判定一“跳”的条件（相邻两句）：
  出现新地点词，且没有换场标记（第二天/后来/过了…），且两句实体无交集，且词汇重合低于阈值。
"""
import argparse, json, os, re, statistics, sys

CJK = re.compile(r"[\u4e00-\u9fff]")
PLACES = ["屋", "房间", "街", "厂", "车间", "站台", "车站", "河", "桥", "店", "铺子", "楼",
          "院子", "门口", "路上", "城里", "村子", "房", "厅", "厨房", "走廊", "楼梯", "花园",
          "码头", "广场", "雪地", "窗前", "床", "桌子", "长桌", "验灯间"]
MARKERS = ["第二天", "次日", "第二天一早", "天亮", "傍晚", "夜里", "半夜", "过了", "后来",
           "三天后", "一周后", "回到家", "回到", "到了", "那年", "冬天", "春天", "夏天", "秋天",
           "这时", "这时候", "与此同时", "等", "第二天下午", "第二天早晨"]
PRON = ["他", "她", "它", "他们", "她们"]


def cjk(s):
    return len(CJK.findall(s))


def sents(t):
    return [x.strip() for x in re.split(r"(?<=[。！？])", t) if x.strip()]


def names(t):
    out = []
    for m in re.finditer(r"([\u4e00-\u9fff]{2,3})(?=说|问|道|笑|答|喊|叫|看|走|坐|站|想)", t):
        c = m.group(1)
        if not any(x in c for x in ("医生", "工头", "老人", "女人", "男人", "孩子", "少年")):
            out.append(c)
    return out


def ents(s):
    e = set(names(s))
    e |= {p for p in PRON if p in s}
    e |= {p for p in PLACES if p in s}
    return e


def bigrams(s):
    s = re.sub(r"[^\u4e00-\u9fff]", "", s)
    return set(s[i:i + 2] for i in range(len(s) - 1))


def analyze(t):
    ss = sents(t)
    if len(ss) < 3:
        return None
    jumps = 0
    pairs = 0
    coh = []
    for i in range(1, len(ss)):
        a, b = ss[i - 1], ss[i]
        pairs += 1
        ea, eb = ents(a), ents(b)
        ba, bb = bigrams(a), bigrams(b)
        ov = len(ba & bb) / max(1, len(ba | bb))
        coh.append(ov)
        new_place = bool({p for p in PLACES if p in b} - {p for p in PLACES if p in a})
        marked = any(b.startswith(m) or m in b[:8] for m in MARKERS)
        ent_share = bool(ea & eb)
        if new_place and not marked and not ent_share and ov < 0.06:
            jumps += 1
    return {"chars": cjk(t), "pairs": pairs, "jumps": jumps,
            "jump_rate": round(jumps / max(1, pairs), 3),
            "cohesion": round(statistics.mean(coh), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl")
    ap.add_argument("--field", default="out")
    ap.add_argument("--file")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    texts = []
    if args.jsonl:
        for line in open(args.jsonl, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get(args.field):
                    texts.append(r[args.field])
    if args.file:
        raw = open(args.file, encoding="utf-8").read()
        ps = [p.strip() for p in raw.split("\n\n") if p.strip()]
        texts += ["\n".join(ps[i:i + 2]) for i in range(0, len(ps), 2)]
    rows = [analyze(t) for t in texts]
    rows = [r for r in rows if r]
    if not rows:
        sys.exit("no text")
    print(f"tag={args.tag} n={len(rows)} "
          f"jump_rate={statistics.mean(r['jump_rate'] for r in rows):.3f} "
          f"cohesion={statistics.mean(r['cohesion'] for r in rows):.3f} "
          f"jumps/篇={statistics.mean(r['jumps'] for r in rows):.2f}")


if __name__ == "__main__":
    main()
