# -*- coding: utf-8 -*-
"""Corporate/academic 21-sample grouped bar chart.

Data are the exact model-only outputs in final21_metrics.json. Error bars are
95% bootstrap CIs for means; 2k rate uses Wilson intervals. The chart is drawn
with a deterministic plotting backend (not an image-generation model) so every
number is exact. Visual language follows the installed space-chart-image
"corporate" style.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

BASE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(BASE, "final21_metrics.json"), encoding="utf-8"))
KEYS = ["base", "v01", "v02"]
LABELS = ["基座", "v0.1", "v0.2"]
COLORS = ["#8A8F98", "#2F5D8A", "#C6A15B"]
INK = "#1E2430"
MUTED = "#5C6573"
GRID = "#D9D3C6"
BG = "#FBF9F4"

# Fonts: academic serif for Chinese, Times-like numerals.
_windir = os.environ.get("WINDIR", "")
_candidates = [
    os.path.join(_windir, "Fonts", "msyh.ttc") if _windir else "",
    os.path.join(_windir, "Fonts", "simsun.ttc") if _windir else "",
]
serif = next((p for p in _candidates if p and os.path.exists(p)), "")
fp_cn = FontProperties(fname=serif) if serif else FontProperties(family="serif")
fp_num = FontProperties(family="Times New Roman")
plt.rcParams.update({
    "axes.unicode_minus": False,
    "font.family": fp_cn.get_name() if os.path.exists(serif) else "serif",
    "font.size": 9.5,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#B9B2A4",
    "axes.linewidth": 0.8,
    "xtick.color": INK,
    "ytick.color": MUTED,
    "text.color": INK,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
})
rng = np.random.default_rng(20261107)

n = 21
rows = {k: d[k]["rows"] for k in KEYS}
metrics = ["cjk", "rep4", "dup", "distinct4"]
vals = {k: {m: np.array([r[m] for r in rows[k]], dtype=float) for m in metrics} for k in KEYS}

def boot_ci(x, B=5000):
    idx = rng.integers(0, len(x), size=(B, len(x)))
    means = x[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])

def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)

# Paired bootstrap CI for relative change of means.
def rel_ci(a, b, B=5000):
    idx = rng.integers(0, len(a), size=(B, len(a)))
    ra = (b[idx].mean(axis=1) / a[idx].mean(axis=1) - 1) * 100
    return np.percentile(ra, [2.5, 97.5])

fig = plt.figure(figsize=(16, 9), dpi=200)
gs = fig.add_gridspec(2, 3, left=0.065, right=0.965, top=0.82, bottom=0.25,
                      hspace=0.52, wspace=0.28)

# Header
fig.text(0.065, 0.945, "Fluorestsentsiya-3B v0.2 评测", fontsize=22, fontweight="bold",
         fontproperties=fp_cn, color=INK, ha="left")
fig.text(0.065, 0.912,
         "基座 / v0.1 / v0.2 模型自身输出 · 21 条 held-out · 同一截断与评测脚本",
         fontsize=10.5, fontproperties=fp_cn, color=MUTED, ha="left")
fig.add_artist(Line2D([0.065, 0.965], [0.885, 0.885], color=COLORS[2], linewidth=1.6))
fig.text(0.965, 0.912, "n = 21", fontsize=10.5, fontproperties=fp_num, color=MUTED, ha="right")
fig.text(0.965, 0.945, "v0.2-final", fontsize=11, fontproperties=fp_num, color=COLORS[2], ha="right")

axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
for ax in axs:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B9B2A4")
    ax.spines["bottom"].set_color("#B9B2A4")
    ax.tick_params(axis="both", length=3, colors=MUTED, labelsize=9)
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_fontproperties(fp_cn if lb.get_text().strip() and not lb.get_text().strip().replace(".", "").replace("%", "").isdigit() else fp_num)

def draw_bars(ax, metric, title, fmt, ylim):
    means, errs = [], []
    for k in KEYS:
        x = vals[k][metric]
        lo, hi = boot_ci(x)
        means.append(x.mean())
        errs.append([x.mean() - lo, hi - x.mean()])
    xs = np.arange(len(KEYS))
    bars = ax.bar(xs, means, width=0.58, color=COLORS, edgecolor="white", linewidth=0.8, zorder=3)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, v, fmt.format(v), ha="center", va="bottom",
                fontsize=8.8, fontproperties=fp_num, color=INK, zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels(LABELS, fontproperties=fp_cn, fontsize=9.5)
    ax.set_ylim(*ylim)
    ax.set_title(title, loc="left", pad=9, fontproperties=fp_cn)
    ax.set_xlim(-0.55, 2.55)

draw_bars(axs[0], "cjk", "平均汉字数", "{:.0f}", (0, 1900))
draw_bars(axs[2], "rep4", "4-gram 重复率 rep4", "{:.4f}", (0, 0.30))
draw_bars(axs[3], "dup", "重复句率", "{:.2%}", (0, 0.21))
draw_bars(axs[4], "distinct4", "distinct4", "{:.4f}", (0, 1.02))

# 2k rate: Wilson intervals
ax = axs[1]
ks = [d[k]["two_k"] for k in KEYS]
ps = [k / n for k in ks]
xs = np.arange(len(KEYS))
bars = ax.bar(xs, ps, width=0.58, color=COLORS, edgecolor="white", linewidth=0.8, zorder=3)
for i, (k, p) in enumerate(zip(ks, ps)):
    ax.text(i, p, f"{k}/21", ha="center", va="bottom", fontsize=8.8,
            fontproperties=fp_num, color=INK, zorder=5)
ax.set_xticks(xs)
ax.set_xticklabels(LABELS, fontproperties=fp_cn, fontsize=9.5)
ax.set_ylim(0, 0.72)
ax.set_title("2,000 字达成率", loc="left", pad=9, fontproperties=fp_cn)
ax.set_xlim(-0.55, 2.55)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

# Relative change, paired bootstrap
ax = axs[5]
chg = []
err = []
for m in ["cjk", "rep4", "dup", "distinct4"]:
    a, b = vals["v01"][m], vals["v02"][m]
    c = (b.mean() / a.mean() - 1) * 100
    lo, hi = rel_ci(a, b)
    chg.append(c)
    err.append([c - lo, hi - c])
labels = ["平均汉字数", "rep4", "重复句率", "distinct4"]
colors = [COLORS[1] if x >= 0 else "#A44A3F" for x in chg]
# semantic colors: for repetition metrics, negative is the desired direction
colors = []
for lab, x in zip(labels, chg):
    if lab in ("rep4", "重复句率"):
        colors.append("#4F7A5A" if x <= 0 else "#A44A3F")
    else:
        colors.append("#4F7A5A" if x >= 0 else "#A44A3F")
bars = ax.bar(range(4), chg, width=0.55, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
ax.axhline(0, color="#3B424E", linewidth=0.9, zorder=2)
ax.axhline(20, color="#4F7A5A", linestyle=(0, (3, 2)), linewidth=1.0, alpha=0.8, zorder=1)
ax.axhline(-20, color="#4F7A5A", linestyle=(0, (3, 2)), linewidth=1.0, alpha=0.8, zorder=1)
for i, (v, e) in enumerate(zip(chg, err)):
    ax.text(i, v + (4 if v >= 0 else -4), f"{v:+.1f}%", ha="center",
            va="bottom" if v >= 0 else "top", fontsize=8.8, fontproperties=fp_num, color=INK, zorder=5)
ax.set_xticks(range(4))
ax.set_xticklabels(labels, fontproperties=fp_cn, fontsize=9.5)
ax.set_ylim(-100, 130)
ax.set_title("v0.2 相对 v0.1 的变化", loc="left", pad=9, fontproperties=fp_cn)
ax.set_ylabel("相对变化（%）", fontproperties=fp_cn, fontsize=9, color=MUTED)
# ±20% reference lines are described in the footnote; no inline clutter.

# Legend
handles = [Line2D([0], [0], marker="s", linestyle="", markersize=9, markerfacecolor=c,
                  markeredgecolor="white", label=l) for c, l in zip(COLORS, LABELS)]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.965, 0.875), ncol=3,
           frameon=False, prop=fp_cn, fontsize=9.5, handletextpad=0.5, columnspacing=1.3)

# Footnotes / metric definitions, academic
note = (
    "指标定义与评估口径（n = 21；同一 held-out 集、同一截断与评测脚本）  "
    "平均汉字数：每条输出在 2,000 个汉字处按句界截断后取汉字数均值；"
    "2,000 字达成率：截断后汉字数 ≥ 1,800 的样本比例；"
    "rep4：4-gram 重复率 = (总 4-gram 数 - 唯一 4-gram 数) / 总 4-gram 数，数值降低表示重复减少；"
    "重复句率：完全重复句数 / 总句数，数值降低表示重复句减少；"
    "distinct4：唯一 4-gram 数 / 总 4-gram 数，数值增大表示 4-gram 多样性增加。\n"
    "相对变化 = ((v0.2 - v0.1) / v0.1) × 100%；复读类指标的负值表示改善；"
    "相对变化图中的虚线为 ±20% 参考线。"
    "图中数值为 21 条样本的均值或比例；逐条数据见 final21_metrics.json。"
    "解码：基座与 v0.1 使用 repeat_penalty = 1.1；v0.2 使用 DRY multiplier = 0.8、repeat_penalty = 1.15；max_tokens = 2,400。"
)
fig.text(0.065, 0.19, note, fontsize=7.1, fontproperties=fp_cn, color=MUTED,
         ha="left", va="top", linespacing=1.55, wrap=True)
fig.text(0.065, 0.055, "数据：final21_metrics.json · 模型自身输出，无人工改稿 · 2026-11-07",
         fontsize=7.4, fontproperties=fp_cn, color="#8B8578", ha="left")
fig.text(0.965, 0.055, "space-chart-image · corporate",
         fontsize=7.4, fontproperties=fp_num, color="#8B8578", ha="right")

out = os.path.join(BASE, "final21_bar_corporate.png")
fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=BG)
print(out, os.path.getsize(out))
