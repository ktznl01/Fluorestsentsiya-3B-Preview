# -*- coding: utf-8 -*-
"""把用户点名的七类毛病量成指标：句长、重复、指代、实体衔接、首尾呼应。

用法：
  python story_metrics.py --jsonl bench/quality/endcard_scaled.jsonl --field out
  python story_metrics.py --file 正文.md
"""
import argparse, json, re, statistics, sys
from collections import Counter

CJK = re.compile(r"[\u4e00-\u9fff]")
SURNAME = set("赵钱孙李周吴郑王冯陈蒋沈韩杨朱秦许何吕施张孔曹严华金魏陶姜谢邹喻柏窦章苏潘葛范彭郎鲁韦马苗方俞任袁柳鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常于傅卞齐康伍余元顾孟平黄穆萧尹姚邵汪祁毛狄米贝明臧计伏戴谈宋茅庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄江童颜郭梅盛林钟徐邱骆高夏蔡田樊胡凌霍虞万支柯管卢莫房裘应宗丁邓郁单杭洪包左石崔龚程邢裴陆荣翁荀羊惠甄")
PRONOUN = ("他", "她", "它", "他们", "她们", "自己", "对方")


def sentences(t):
    return [s.strip() for s in re.split(r"(?<=[。！？])", t) if s.strip()]


ROLE = ("医生", "工头", "老人", "女人", "男人", "孩子", "少年", "母亲", "父亲", "先生",
        "太太", "小姐", "公爵", "伯爵", "将军", "神父", "掌柜", "车夫", "仆人", "警察")


def names(t):
    """实体：出现在“说/问/道”前、且在全文中至少出现两次的 2–3 字串（兼容外国人名）。"""
    cand = Counter()
    for m in re.finditer(r"([\u4e00-\u9fff]{2,3})(?=说|问|道|笑|答|喊|叫)", t):
        cand[m.group(1)] += 1
    out = [c for c, k in cand.items() if k >= 2 and not any(x in c for x in ROLE)]
    return out


def analyze(t):
    t = t or ""
    ss = sentences(t)
    if len(ss) < 3:
        return None
    lens = [len(CJK.findall(s)) for s in ss]
    body = re.sub(r"\s+", "", t)
    n = len(body)
    grams = [body[i:i + 4] for i in range(max(0, n - 3))]
    gc = Counter(grams)
    rep4 = sum(v - 1 for v in gc.values() if v > 1) / max(1, len(grams))
    top_rep = gc.most_common(1)[0][1] if gc else 0

    paras = [p for p in re.split(r"\n\s*\n|\n", t) if p.strip()]
    ent_para = [set(names(p)) for p in paras]
    link = 0
    for i in range(1, len(ent_para)):
        if ent_para[i] and ent_para[i - 1] and (ent_para[i] & ent_para[i - 1]):
            link += 1
    link_rate = link / max(1, len(paras) - 1)

    def bigrams(x):
        x = re.sub(r"[^\u4e00-\u9fff]", "", x)
        return set(x[i:i + 2] for i in range(len(x) - 1))
    adj = []
    for i in range(1, len(ss)):
        a, b = bigrams(ss[i - 1]), bigrams(ss[i])
        if a and b:
            adj.append(len(a & b) / len(a | b))
    adj_overlap = statistics.mean(adj) if adj else 0.0

    first, last = ss[0], ss[-1]
    big = lambda s: set(s[i:i + 2] for i in range(len(s) - 1))
    echo = len(big(first) & big(last)) / max(1, len(big(first)))

    pron = sum(body.count(w) for w in PRONOUN)
    nm = names(t)
    return {
        "chars": n,
        "sent_mean": round(statistics.mean(lens), 1),
        "short_pct": round(sum(1 for x in lens if x < 10) / len(lens) * 100),
        "long_pct": round(sum(1 for x in lens if x >= 40) / len(lens) * 100),
        "sent_max": max(lens),
        "rep4": round(rep4, 3),
        "rep_top": top_rep,
        "open_uniq": round(len({s[:2] for s in ss}) / len(ss), 2),
        "ent_link": round(link_rate, 2),
        "adj_overlap": round(adj_overlap, 3),
        "pron_per_k": round(pron / max(1, n) * 1000, 1),
        "name_uniq": len(set(nm)),
        "echo": round(echo, 2),
    }


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
        texts += ["\n\n".join(ps[i:i + 2]) for i in range(0, len(ps), 2)]
    rows = [analyze(t) for t in texts]
    rows = [r for r in rows if r]
    if not rows:
        sys.exit("no text")
    keys = ["sent_mean", "short_pct", "long_pct", "sent_max", "rep4", "rep_top",
            "open_uniq", "adj_overlap", "pron_per_k", "name_uniq", "echo"]
    print(("tag=" + args.tag + " ") if args.tag else "", "n=", len(rows))
    for k in keys:
        print(f"  {k}: {statistics.mean(r[k] for r in rows):.2f}")


if __name__ == "__main__":
    main()
