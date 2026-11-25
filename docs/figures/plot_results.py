"""Rebuild the two README figures with Matplotlib and a Chinese font.

Run python docs/figures/plot_results.py --font /path/to/a/CJK-font.ttc
or omit --font on a system with Noto Sans CJK SC installed.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = Path(__file__).resolve().parent
DATA = OUT.parents[1] / "verification"
INK, MUTED, GRID = "#182F45", "#4B6172", "#DCE4EA"
TORCH, JITTOR, PAPER = "#AD4E20", "#216AA4", "#728394"


def heading(fig, title, subtitle):
    fig.text(.055, .965, title, va="top", size=23, weight="bold", color=INK)
    fig.text(.055, .895, subtitle, va="top", size=12, color=MUTED)


def epoch_curves():
    with (DATA / "coco20k-epoch-ap.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.3))
    fig.subplots_adjust(left=.085, right=.965, bottom=.13, top=.755, wspace=.22, hspace=.54)
    heading(fig, "四阶段训练的验证 AP", "COCO 20k 训练  /  val2017 逐轮验证  /  AP50–95（点）")
    panels = [
        ("TP", "TP  文本提示检测", [10, 20, 30, 40], (10, 44)),
        ("VP", "VP  视觉提示检测", [10, 20, 30], (5, 35)),
        ("PF", "PF  单类检测", [44.5, 45.0, 45.5], (44.4, 45.6)),
        ("SEG", "SEG  实例分割（掩码）", [20, 24, 28, 32], (20, 32)),
    ]
    for ax, (stage, title, yticks, ylim) in zip(axes.flat, panels):
        points = [r for r in rows if r["stage"] == stage]
        epochs = [int(r["epoch"]) for r in points]
        expected = 30 if stage == "TP" else 10
        if epochs != list(range(1, expected + 1)):
            raise ValueError(f"Incomplete epoch series for {stage}")
        ax.set_title(title, loc="left", pad=8, fontsize=14, color=INK, fontweight="bold")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=.8)
        for key, color, marker, style in [
            ("jittor", JITTOR, "o", "-"), ("pytorch", TORCH, "s", "--")
        ]:
            ax.plot(epochs, [float(r[key + "_ap50_95"]) for r in points],
                    label="Jittor" if key == "jittor" else "PyTorch",
                    color=color, linestyle=style, linewidth=2.1, marker=marker,
                    markersize=4, markerfacecolor="white", markeredgewidth=1.2,
                    markevery=3 if stage == "TP" else 1)
        ax.set_xlim(.65, expected + .35)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30] if stage == "TP" else [1, 2, 4, 6, 8, 10])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(.97, .875),
               ncol=2, frameon=False, handlelength=2.4, columnspacing=1.3)
    fig.text(.52, .035, "训练轮次（epoch）", color=MUTED, ha="center", fontsize=12)
    fig.savefig(OUT / "coco-training.png", dpi=170)
    plt.close(fig)


def lvis_benchmarks():
    official = json.loads((DATA / "yoloe26-official-lvis.json").read_text(encoding="utf-8"))
    trained = json.loads((DATA / "lvis-coco20k-release.json").read_text(encoding="utf-8"))
    ap = official["metrics"]["AP"]
    # Historical non-E2E values from YOLOE Table 1, text-prompt column:
    # https://arxiv.org/html/2503.07465v1#S4.T1
    groups = [
        ("YOLOE 原论文  ·  非端到端", [
            ("YOLOE-v8s", 27.9, PAPER, 1), ("YOLOE-11s", 27.5, PAPER, 1)]),
        ("YOLOE-26s  ·  端到端", [
            ("技术报告", official["paper_e2e_ap"], PAPER, 1),
            ("官方权重  /  PyTorch", ap["pytorch"], TORCH, 3),
            ("官方权重  /  Jittor", ap["jittor"], JITTOR, 3)]),
        ("YOLOE-26s  ·  COCO 20k 自训，端到端", [
            ("自训权重  /  PyTorch", trained["original_pytorch"]["lvis"]["AP"], TORCH, 3),
            ("自训权重  /  Jittor", trained["repaired_native_jittor"]["lvis"]["AP"], JITTOR, 3)]),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.65))
    fig.subplots_adjust(left=.285, right=.945, bottom=.14, top=.815)
    heading(fig, "LVIS 文本提示检测", "LVIS minival  /  1,203 个类别  /  bbox Fixed AP（点）")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=.8)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 36)
    ax.set_xticks(range(0, 36, 5))
    ax.set_xlabel("bbox Fixed AP", labelpad=6, color=MUTED)
    y = 0
    for title, values in groups:
        ax.text(-.35, y, title, transform=ax.get_yaxis_transform(), va="center",
                fontweight="bold", fontsize=13, color=INK)
        for label, value, color, decimals in values:
            y += 1
            ax.text(-.025, y, label, va="center", ha="right",
                    transform=ax.get_yaxis_transform(), fontsize=11, color=MUTED)
            ax.barh(y, value, height=.53, color=color)
            ax.text(value + .4, y, f"{value:.{decimals}f}", va="center", color=INK, fontsize=12)
        y += 1.45
    ax.set_ylim(y - .8, -.55)
    ax.set_yticks([])
    fig.savefig(OUT / "lvis-text-prompt.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, help="CJK .ttf, .otf or .ttc file")
    args = parser.parse_args()
    family = "Noto Sans CJK SC"
    if args.font:
        font_manager.fontManager.addfont(args.font)
        family = font_manager.FontProperties(fname=args.font).get_name()
    font_manager.findfont(family, fallback_to_default=False)
    plt.rcParams.update({"font.family": family, "font.size": 11, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.major.size": 0, "ytick.major.size": 0,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    epoch_curves()
    lvis_benchmarks()
