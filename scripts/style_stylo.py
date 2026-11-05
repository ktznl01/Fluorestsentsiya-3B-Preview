# -*- coding: utf-8 -*-
"""文体学风格保真度：Burrows's Delta。

以人写语料（旧译俄国小说）为作者画像，计算生成文本到画像的 Delta 距离；
越低越接近该风格。同时给出人写留出文本的 Delta 作为参照下界。
"""
import argparse, json, math, os, re, statistics
from paths import CORPUS

CJK = re.compile(r"[\u4e00-\u9fff]")
# 高频功能字（中文的“功能词”近似）
FUNC = list("的了着过是在有就都又还也很把被给对从向和与但而却只才已曾没别再更最太不他她我你它们这那其之为以于而且所以因为如果虽然但是然而于是便却并且")
SENTS = re.compile(r"(?<=[。！？])")


def feats(text):
    n = len(CJK.findall(text))
    if n < 50:
        return None
    f = {}
    for ch in FUNC:
        f[ch] = text.count(ch) / n
    f["_sent_len"] = statistics.mean(len(x) for x in SENTS.split(text) if x.strip()) if SENTS.split(text) else 0
    f["_comma"] = text.count("，") / n
    f["_period"] = text.count("。") / n
    f["_dash"] = (text.count("—") + text.count("——")) / n
    f["_quote"] = (text.count("“") + text.count("”")) / n
    return f


def profile(texts):
    fs = [f for f in (feats(t) for t in texts) if f]
    keys = fs[0].keys()
    mu = {k: statistics.mean(f[k] for f in fs) for k in keys}
    sd = {k: (statistics.pstdev(f[k] for f in fs) or 1e-6) for k in keys}
    return mu, sd


def delta(f, mu, sd):
    return statistics.mean(abs(f[k] - mu[k]) / sd[k] for k in mu)


def load_corpus(path, start, minlen=400, maxlen=1200):
    ps = [p.strip() for p in open(path, encoding="utf-8").read().split("\n\n")[start:] if p.strip()]
    return [p for p in ps if minlen <= len(CJK.findall(p)) <= maxlen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(CORPUS / "main.txt"))
    ap.add_argument("--samples", required=True, help="jsonl，字段 text")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--func-only", action="store_true")
    args = ap.parse_args()

    human = load_corpus(args.corpus, args.start)
    split = int(len(human) * 0.8)
    train, held = human[:split], human[split:]
    mu, sd = profile(train)
    held_d = [delta(f, mu, sd) for f in (feats(t) for t in held) if f]
    if args.func_only:
        keep = [k for k in mu if not k.startswith("_")]
        mu = {k: mu[k] for k in keep}
        sd = {k: sd[k] for k in keep}
    gen = [json.loads(l)["text"] for l in open(args.samples, encoding="utf-8") if l.strip()]
    gen_d = [delta(f, mu, sd) for f in (feats(t) for t in gen) if f]
    out = {"tag": args.tag,
           "human_heldout_delta": round(statistics.mean(held_d), 3),
           "model_delta": round(statistics.mean(gen_d), 3) if gen_d else None,
           "n_model": len(gen_d),
           "fidelity": round(statistics.mean(held_d) / statistics.mean(gen_d), 3) if gen_d else None}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
