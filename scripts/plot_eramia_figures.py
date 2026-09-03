# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np

def configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "axes.titleweight": "bold",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "figure.titlesize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.6,
    })

MODEL_COLORS = {
    "Lexical": "#1b9e77",
    "Qwen3.5-2B": "#d95f02",
    "Qwen3.5-0.8B": "#7570b3",
    "SmolLM2-1.7B": "#e7298a",
    "TinyLlama-1.1B": "#66a61e",
    "Qwen2.5-0.5B": "#386cb0",
    "Qwen2.5-0.5B + LoRA": "#f781bf",
    "Always Abstain": "#999999",
}

CASCADE_COLORS = [
    "#e0e0e0",
    "#a6bddb",
    "#74a9cf",
    "#023858",
]
CASCADE_LABELS = [
    "1. Invalid JSON",
    "2. Valid JSON,\nInvalid Contract",
    "3. Valid Contract,\nWrong Decision",
    "4. Valid Contract\n& Exact Match",
]

CALL_COLOR = "#1b9e77"
ABSTAIN_COLOR = "#d95f02"

PARETO_DATA = [
    {"name": "Lexical", "x": 100.0, "y": 77.40, "recall": 54.79, "json_valid": 100.0, "note": ""},
    {"name": "Qwen3.5-2B", "x": 100.0, "y": 50.00, "recall": 0.0, "json_valid": 100.0, "note": "degenerate\nabstention policy"},
    {"name": "Qwen3.5-0.8B", "x": 60.15, "y": 43.60, "recall": 33.45, "json_valid": 100.0, "note": ""},
    {"name": "SmolLM2-1.7B", "x": 0.0, "y": 0.0, "recall": 0.0, "json_valid": 96.5, "note": "96.5% valid JSON\nbut 0% contract"},
    {"name": "TinyLlama-1.1B", "x": 0.0, "y": 0.0, "recall": 0.0, "json_valid": 43.4, "note": "43.4% valid JSON"},
    {"name": "Qwen2.5-0.5B", "x": 0.0, "y": 0.0, "recall": 0.0, "json_valid": 70.2, "note": "70.2% valid JSON"},
]
LORA_DATA = {
    "name": "Qwen2.5-0.5B + LoRA",
    "x": 100.0,
    "y": 100.0,
    "recall": 100.0,
    "json_valid": 100.0,
    "is_lora": True,
    "note": "supervised LoRA\n(non-zero-shot)",
}

CASCADE_DATA = [
    ("Lexical", 100.00, 100.00, 77.40),
    ("Qwen3.5-2B", 100.00, 100.00, 50.00),
    ("Qwen3.5-0.8B", 100.00, 60.15, 43.60),
    ("SmolLM2-1.7B", 96.47, 0.00, 0.00),
    ("TinyLlama-1.1B", 43.42, 0.00, 0.00),
    ("Qwen2.5-0.5B", 70.17, 0.00, 0.00),
]

LEAKAGE_DATA = {
    "Lexical": {"official": 77.92, "phrase": 77.40},
    "Qwen3.5-0.8B": {"official": 54.65, "phrase": 43.60},
    "Qwen3.5-2B": {"official": 50.98, "phrase": 50.00},
}

CONFUSION_DATA = {
    "Ground Truth": {"call": 50.00, "abstain": 50.00},
    "Lexical": {"call": 27.40, "abstain": 72.60},
    "Qwen3.5-0.8B": {"call": 73.13, "abstain": 26.87},
    "Qwen3.5-2B": {"call": 0.00, "abstain": 100.00},
}

def _ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def _save(fig, outdir: Path, basename: str, dpi: int = 300) -> None:
    pdf = outdir / f"{basename}.pdf"
    png = outdir / f"{basename}.png"
    fig.savefig(str(pdf), format="pdf", dpi=dpi)
    fig.savefig(str(png), format="png", dpi=dpi)
    print(f"[ok] {pdf}  +  {png}")
    plt.close(fig)

def plot_pareto_frontier(outdir: Path, dpi: int = 300, with_lora: bool = True) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(6.8, 5.15))

    ax.set_xlim(-6, 106)
    ax.set_ylim(-6, 106)
    ax.set_xlabel("Contract Validity (%)  ->", fontweight="bold")
    ax.set_ylabel("Exact Match (%)  ->", fontweight="bold")
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, which="major", color="#e6e6e6", linestyle="-", linewidth=0.6, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)

    ax.add_patch(plt.Rectangle((-6, -6), 56, 56, facecolor="#fee5d9", alpha=0.55, edgecolor="none", zorder=0))
    ax.add_patch(plt.Rectangle((50, -6), 56, 56, facecolor="#feedde", alpha=0.45, edgecolor="none", zorder=0))
    ax.add_patch(plt.Rectangle((50, 50), 56, 56, facecolor="#e5f5e0", alpha=0.55, edgecolor="none", zorder=0))
    ax.add_patch(plt.Rectangle((-6, 50), 56, 56, facecolor="#f0f0f0", alpha=0.35, edgecolor="none", zorder=0))

    ax.axvline(50, color="#999999", linestyle=":", linewidth=1.0, alpha=0.7, zorder=1)
    ax.axhline(50, color="#666666", linestyle="--", linewidth=1.35, alpha=0.95, zorder=1)

    ax.text(
        52, 51.8, "Baseline: Always Abstain  (50% EM)", fontsize=7.2, color="#333333",
        va="bottom", ha="left", style="italic",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#cccccc", alpha=0.95),
        zorder=5,
    )

    ax.text(25, 25, "Structural failure\n(contract invalid)", ha="center", va="center", fontsize=7, color="#7f2704", alpha=0.85, style="italic", zorder=2)
    ax.text(78, 25, "Conservative\n(valid but\nover-abstaining)", ha="center", va="center", fontsize=7, color="#7f2704", alpha=0.85, style="italic", zorder=2)
    ax.text(78, 78, "Ideal quadrant\n(high validity\n+ high accuracy)", ha="center", va="center", fontsize=7.5, color="#00441b", alpha=0.9, style="italic", weight="bold", zorder=2)
    ax.text(25, 78, "Unreachable\nin zero-shot", ha="center", va="center", fontsize=6.5, color="#525252", alpha=0.7, style="italic", zorder=2)

    frontier_x = [60.15, 100.0]
    frontier_y = [43.60, 77.40]
    ax.plot(frontier_x, frontier_y, color="#525252", linestyle=(0, (4, 3)), linewidth=1.2, alpha=0.85, zorder=3)
    ax.scatter(frontier_x, frontier_y, s=70, facecolors="none", edgecolors="#525252", linewidths=1.1, zorder=4, alpha=0.9)

    ax.annotate(
        "Empirical trade-off\n(zero-shot frontier)", xy=(80, 60), xytext=(38, 88),
        ha="center", va="center", fontsize=6.8, color="#252525",
        arrowprops=dict(arrowstyle="-|>", color="#525252", lw=1.0, connectionstyle="arc3,rad=0.18"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#bdbdbd", alpha=0.96),
        zorder=6,
    )
    if with_lora:
        ax.annotate(
            "", xy=(100, 100), xytext=(100, 84),
            arrowprops=dict(arrowstyle="-|>", color="#c51b7d", lw=1.1, linestyle="--"),
            zorder=6,
        )

    def marker_size(recall: float) -> float:
        return 75 + 445 * (recall / 100.0)

    jitter = {
        "SmolLM2-1.7B": (-2.6, 2.2),
        "TinyLlama-1.1B": (0.0, -2.7),
        "Qwen2.5-0.5B": (2.4, 0.9),
    }

    all_data = PARETO_DATA.copy()
    if with_lora:
        all_data.append(LORA_DATA)

    for d in all_data:
        name = d["name"]
        x, y, rec = d["x"], d["y"], d["recall"]
        is_lora = d.get("is_lora", False)

        dx, dy = jitter.get(name, (0, 0))
        px, py = x + dx, y + dy

        s = marker_size(rec)
        color = MODEL_COLORS.get(name, "#333333")
        edgec = "white" if not is_lora else "#c51b7d"
        lw = 1.0 if not is_lora else 1.6
        marker = "o" if not is_lora else "*"
        if is_lora:
            s = marker_size(rec) * 1.35
            marker = "*"

        ax.scatter(px, py, s=s + 45, c="black", alpha=0.07, linewidths=0, zorder=3, marker=marker)
        ax.scatter(px, py, s=s, c=color, edgecolors=edgec, linewidths=lw, zorder=5, marker=marker, alpha=0.97)

        offsets = {
            "Lexical": (14, 8),
            "Qwen3.5-2B": (14, -14),
            "Qwen3.5-0.8B": (14, -16),
            "SmolLM2-1.7B": (18, 26),
            "Qwen2.5-0.5B": (24, 15),
            "TinyLlama-1.1B": (28, 4),
            "Qwen2.5-0.5B + LoRA": (-36, -8),
        }
        ox, oy = offsets.get(name, (12, 12))

        if name == "Lexical":
            label = "Lexical Control\n(100.0, 77.4)  R$_{call}^{exact}$=54.8%"
        elif name == "Qwen3.5-2B":
            label = "Qwen3.5-2B\n(100.0, 50.0)  R$_{call}^{exact}$=0.0%\n\u2020 degenerate abstention"
        elif name == "Qwen3.5-0.8B":
            label = "Qwen3.5-0.8B\n(60.2, 43.6)  R$_{call}^{exact}$=33.5%"
        elif name == "Qwen2.5-0.5B + LoRA":
            label = "Qwen2.5-0.5B + LoRA *\n(100.0, 100.0)  R$_{call}^{exact}$=100%\n[supervised adaptor reference]"
        else:
            label = f"{name}\n(0.0, 0.0)  R$_{{call}}^{{exact}}$=0%\n({d['json_valid']:.1f}% valid JSON)"

        box_face = "white" if not is_lora else "#fff7fb"
        box_edge = color if not is_lora else "#c51b7d"
        fsz = 6.4 if name in jitter else (6.5 if is_lora else 6.7)

        ax.annotate(
            label,
            xy=(px, py),
            xytext=(px + ox, py + oy),
            ha="left" if ox >= 0 else "right",
            va="center",
            fontsize=fsz,
            color="#1a1a1a",
            bbox=dict(boxstyle="round,pad=0.28", facecolor=box_face, edgecolor=box_edge, alpha=0.97, linewidth=0.9),
            arrowprops=dict(arrowstyle="-|>", color=box_edge, lw=0.9, shrinkA=2, shrinkB=3, connectionstyle="angle3,angleA=0,angleB=90"),
            zorder=7,
        )

    legend_sizes = [0, 33.45, 54.79, 100]
    size_handles = []
    for rs in legend_sizes:
        s = marker_size(rs)
        h = plt.scatter([], [], s=s, c="#525252", edgecolors="white", linewidths=0.8, alpha=0.95)
        size_handles.append(h)
    size_labels = ["0%", "33.5%", "54.8%", "100%"]
    leg1 = ax.legend(
        size_handles, size_labels,
        title="Marker Size \u221d Exact Call Recall",
        loc="lower right", bbox_to_anchor=(0.98, 0.02),
        frameon=True, fancybox=True, shadow=False,
        handletextpad=0.8, borderpad=0.6, labelspacing=1.0,
        facecolor="white", edgecolor="#bdbdbd",
    )
    leg1.get_title().set_fontsize(7)
    ax.add_artist(leg1)

    if with_lora:
        lora_handle = Line2D([0], [0], marker="*", color="w", markerfacecolor=MODEL_COLORS["Qwen2.5-0.5B + LoRA"],
                             markeredgecolor="#c51b7d", markersize=11, markeredgewidth=1.2, linestyle="None")
        ax.legend([lora_handle], ["Supervised LoRA reference"], loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, facecolor="#fff7fb", edgecolor="#c51b7d", handletextpad=0.4)

    ax.text(
        0.5, -5.2, "\u2020 Three models at (0,0) jittered for visual clarity \u2014 all exhibit 0% contract validity (structural failure).",
        ha="left", va="top", fontsize=6.2, color="#525252", style="italic",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f7f7f7", edgecolor="#d9d9d9", alpha=0.95),
    )

    ax.set_title("Trade-off Space: Contract Validity vs. Exact Match (Phrase-Disjoint)\nMarker area proportional to Exact Call Recall", fontsize=8.5, pad=10, loc="center")

    fig.tight_layout(pad=0.6)
    _save(fig, outdir, "fig_pareto_frontier", dpi=dpi)

def plot_structural_cascade(outdir: Path, dpi: int = 300) -> None:
    configure_style()
    rows = []
    for name, parse, valid, exact in CASCADE_DATA:
        s1 = max(0, 100.0 - parse)
        s2 = max(0, parse - valid)
        s3 = max(0, valid - exact)
        s4 = max(0, exact)
        total = s1 + s2 + s3 + s4
        if abs(total - 100) > 0.01:
            factor = 100 / total
            s1, s2, s3, s4 = s1 * factor, s2 * factor, s3 * factor, s4 * factor
        rows.append((name, [s1, s2, s3, s4]))

    n = len(rows)
    fig_h = 0.68 * n + 1.8
    fig, ax = plt.subplots(figsize=(7.2, fig_h))

    y_pos = np.arange(n)
    bar_h = 0.58
    lefts = np.zeros(n)

    for idx_stage in range(4):
        widths = np.array([r[1][idx_stage] for r in rows])
        color = CASCADE_COLORS[idx_stage]
        ax.barh(y_pos, widths, left=lefts, height=bar_h, color=color, edgecolor="white", linewidth=0.9, zorder=3)

        for i, (w, l) in enumerate(zip(widths, lefts)):
            if w >= 8.0:
                cx = l + w / 2
                txt_color = "white" if idx_stage >= 2 else "#1a1a1a"
                if idx_stage == 0 or idx_stage == 1:
                    txt_color = "#1a1a1a"
                if idx_stage == 3:
                    txt_color = "white"
                ax.text(
                    cx, y_pos[i], f"{w:.1f}%",
                    ha="center", va="center", fontsize=7, color=txt_color, weight="bold",
                    zorder=5,
                )
        lefts += widths

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Proportion of predictions in phrase-disjoint test (%)  ->", fontweight="bold", fontsize=8.5)
    ax.grid(axis="x", color="#e5e5e5", linestyle="-", linewidth=0.6, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#bdbdbd")
        spine.set_linewidth(0.8)

    patches = [mpatches.Patch(facecolor=CASCADE_COLORS[i], edgecolor="white", linewidth=0.8, label=CASCADE_LABELS[i]) for i in range(4)]
    ax.legend(
        handles=patches, loc="upper center", bbox_to_anchor=(0.5, 1.14),
        ncol=4, frameon=True, fancybox=True, shadow=False,
        facecolor="white", edgecolor="#bdbdbd", handlelength=1.4, handleheight=1.4, columnspacing=1.2, handletextpad=0.6,
        fontsize=6.8,
    )
    ax.set_title("Structural Failure Cascade: 4-Stage Decomposition of Model Output\n100% = total test predictions per model", fontsize=8.5, pad=26, loc="center")

    ax.axvline(100, color="#525252", linewidth=0.8, alpha=0.6, zorder=2)

    fig.text(
        0.01, 0.01,
        "Stages: 1 Invalid JSON -> 2 Valid JSON but invalid contract -> 3 Valid contract but wrong decision -> 4 Valid contract and exact match. Values shown for slices >= 8%.",
        ha="left", va="bottom", fontsize=6.2, color="#525252", style="italic",
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.90])
    _save(fig, outdir, "fig_structural_cascade", dpi=dpi)

def plot_leakage_and_confusion(outdir: Path, dpi: int = 300) -> None:
    configure_style()
    fig = plt.figure(figsize=(9.2, 4.2), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.32)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    axA.set_title("A \u2014 Template Leakage: Official vs. Phrase-Disjoint\n(Exact Match %)", fontsize=8.5, pad=8, loc="center")
    x = np.array([0, 1])
    x_labels = ["Official\n(speaker-disjoint)", "Phrase-Disjoint\n(template-disjoint)"]

    models_slope = [
        ("Lexical", LEAKAGE_DATA["Lexical"]["official"], LEAKAGE_DATA["Lexical"]["phrase"], "#1b9e77", "o", 1.9, 0),
        ("Qwen3.5-2B", LEAKAGE_DATA["Qwen3.5-2B"]["official"], LEAKAGE_DATA["Qwen3.5-2B"]["phrase"], "#7570b3", "s", 1.8, 0),
        ("Qwen3.5-0.8B", LEAKAGE_DATA["Qwen3.5-0.8B"]["official"], LEAKAGE_DATA["Qwen3.5-0.8B"]["phrase"], "#d73027", "D", 2.6, 1),
    ]

    axA.set_xlim(-0.18, 1.18)
    axA.set_ylim(20, 85)
    axA.set_xticks(x)
    axA.set_xticklabels(x_labels, fontsize=7.5)
    axA.set_ylabel("Exact Match (%)", fontweight="bold", fontsize=8)
    axA.set_yticks([20, 30, 40, 50, 60, 70, 80])
    axA.grid(True, axis="y", color="#e6e6e6", linestyle="-", linewidth=0.6, alpha=0.9, zorder=0)
    axA.set_axisbelow(True)

    gap_poly = Polygon([[0, 54.65], [1, 43.60], [1, 54.65], [0, 54.65]], closed=True, facecolor="#fee5d9", edgecolor="#fc8d59", alpha=0.55, linewidth=0.9, linestyle="--", zorder=1)
    axA.add_patch(gap_poly)

    for name, y_off, y_phr, col, mkr, lw, z in models_slope:
        y_vals = [y_off, y_phr]
        axA.plot(x, y_vals, color=col, linewidth=lw, marker=mkr, markersize=6.5, markerfacecolor=col, markeredgecolor="white", markeredgewidth=0.9, zorder=4 if name != "Qwen3.5-0.8B" else 5, alpha=0.95)
        axA.text(x[0] - 0.04, y_off + (0.7 if name != "Qwen3.5-0.8B" else 1.1), f"{y_off:.2f}%", ha="right", va="center", fontsize=6.8, color=col, weight="bold",
                 bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=col, alpha=0.95, linewidth=0.7))
        axA.text(x[1] + 0.04, y_phr + (0.7 if name != "Lexical" else -1.0), f"{y_phr:.2f}%", ha="left", va="center", fontsize=6.8, color=col, weight="bold",
                 bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=col, alpha=0.95, linewidth=0.7))

    axA.annotate(
        "Drop: \u221211.05 pp\n(template leakage gap)", xy=(0.82, 47.2), xytext=(0.32, 26.5),
        ha="center", va="center", fontsize=6.9, color="#b30000", weight="bold",
        arrowprops=dict(arrowstyle="-|>", color="#b30000", lw=1.1, connectionstyle="arc3,rad=0.22"),
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#fff5f0", edgecolor="#b30000", alpha=0.97),
        zorder=6,
    )
    axA.text(0.5, 78.8, "Lexical: 77.92% \u2192 77.40%  stable (\u22120.52 pp)", ha="center", va="center", fontsize=6.5, color="#1b9e77", style="italic",
             bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#a6dba0", alpha=0.95))
    axA.text(0.5, 52.8, "Qwen3.5-2B: 50.98% \u2192 50.00%  stable", ha="center", va="center", fontsize=6.5, color="#7570b3", style="italic",
             bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#bcbddc", alpha=0.95))

    axA.text(0.0, 21.2, "n phrase-disjoint test = 2,296 (38 clusters)  \u00b7  n official test = 1,934", ha="left", va="center", fontsize=5.8, color="#636363", style="italic")

    # ---------------- Panel B: Action Distribution ----------------
    axB.set_title("B \u2014 Predicted Action (Call vs. Abstain) vs. Ground Truth\n(Phrase-Disjoint Test)", fontsize=8.5, pad=8, loc="center")

    labels = ["Ground\nTruth", "Lexical", "Qwen3.5\n0.8B", "Qwen3.5\n2B"]
    keys = ["Ground Truth", "Lexical", "Qwen3.5-0.8B", "Qwen3.5-2B"]
    xB = np.arange(len(labels))

    call_vals = [CONFUSION_DATA[k]["call"] for k in keys]
    abst_vals = [CONFUSION_DATA[k]["abstain"] for k in keys]

    bar_w = 0.62
    axB.bar(xB, call_vals, width=bar_w, color=["#bdbdbd" if k == "Ground Truth" else CALL_COLOR for k in keys],
            edgecolor="white", linewidth=0.9, zorder=3, label="Call")
    axB.bar(xB, abst_vals, width=bar_w, bottom=call_vals, color=["#f0f0f0" if k == "Ground Truth" else ABSTAIN_COLOR for k in keys],
            edgecolor="white", linewidth=0.9, zorder=3, hatch=["///" if k == "Ground Truth" else "" for k in keys], label="Abstain")

    axB.set_xticks(xB)
    axB.set_xticklabels(labels, fontsize=7.5)
    axB.set_ylabel("Proportion of predictions (%)", fontweight="bold", fontsize=8)
    axB.set_ylim(0, 104)
    axB.set_yticks([0, 25, 50, 75, 100])
    axB.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    axB.grid(True, axis="y", color="#e6e6e6", linestyle="-", linewidth=0.6, alpha=0.9, zorder=0)
    axB.set_axisbelow(True)

    axB.axhline(50, color="#525252", linestyle=":", linewidth=1.0, alpha=0.7, zorder=2)
    axB.text(3.35, 50.8, "50% (Balanced Ground Truth)", ha="right", va="bottom", fontsize=6.2, color="#525252", style="italic")

    for i, (c, a) in enumerate(zip(call_vals, abst_vals)):
        if c >= 8:
            y_c = c / 2
            txt_c = f"{c:.1f}%" if c != 100 else "100%"
            col_c = "white" if keys[i] != "Ground Truth" else "#1a1a1a"
            if c >= 18:
                axB.text(xB[i], y_c + 4, txt_c, ha="center", va="center", fontsize=7, color=col_c, weight="bold", zorder=5)
                axB.text(xB[i], y_c - 4, "Call", ha="center", va="center", fontsize=5.5, color=col_c, style="italic", zorder=5)
            else:
                axB.text(xB[i], y_c, txt_c, ha="center", va="center", fontsize=6.8, color=col_c, weight="bold", zorder=5)
        if a >= 8:
            y_a = c + a / 2
            txt_a = f"{a:.1f}%" if a != 100 else "100%"
            col_a = "white" if (keys[i] != "Ground Truth" and a > 30) or keys[i] == "Qwen3.5-2B" else "#1a1a1a"
            if a >= 18:
                axB.text(xB[i], y_a + 4, txt_a, ha="center", va="center", fontsize=7, color=col_a, weight="bold", zorder=5)
                axB.text(xB[i], y_a - 4, "Abstain", ha="center", va="center", fontsize=5.5, color=col_a, style="italic", zorder=5)
            else:
                axB.text(xB[i], y_a, txt_a, ha="center", va="center", fontsize=6.8, color=col_a, weight="bold", zorder=5)

    axB.annotate(
        "Complete collapse:\n100% abstentions\nR$_{call}^{exact}$=0%", xy=(xB[3], 92), xytext=(xB[2] - 0.15, 88),
        ha="center", va="center", fontsize=6.8, color="#b30000", weight="bold",
        arrowprops=dict(arrowstyle="-|>", color="#b30000", lw=1.1, connectionstyle="angle3,angleA=0,angleB=90"),
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#fff5f0", edgecolor="#b30000", alpha=0.97),
        zorder=6,
    )
    axB.annotate("", xy=(xB[3], 50), xytext=(xB[3] - 0.45, 66), arrowprops=dict(arrowstyle="-|>", color="#b30000", lw=0.9, linestyle="--"), zorder=6)

    axB.text(xB[1], 104.5, "72.6% abst.", ha="center", va="bottom", fontsize=6.2, color=ABSTAIN_COLOR, style="italic")
    axB.text(xB[2], 104.5, "73.1% call", ha="center", va="bottom", fontsize=6.2, color=CALL_COLOR, style="italic")

    handles = [
        mpatches.Patch(facecolor=CALL_COLOR, edgecolor="white", label="Call action"),
        mpatches.Patch(facecolor=ABSTAIN_COLOR, edgecolor="white", label="Abstain policy"),
        mpatches.Patch(facecolor="#f0f0f0", edgecolor="#bdbdbd", hatch="///", label="Ground Truth (50/50 balanced)"),
    ]
    axB.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True, facecolor="white", edgecolor="#bdbdbd", fontsize=6.5, handlelength=1.2)

    for spine in axA.spines.values():
        spine.set_color("#bdbdbd"); spine.set_linewidth(0.8)
    for spine in axB.spines.values():
        spine.set_color("#bdbdbd"); spine.set_linewidth(0.8)

    # Title removed to let LaTeX handle captions cleanly
    _save(fig, outdir, "fig_leakage_and_confusion", dpi=dpi)

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Eramia benchmark figures for SBC LaTeX template")
    parser.add_argument("--outdir", type=str, default="paper/figures", help="Output directory (default: paper/figures)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PNG (default: 300)")
    parser.add_argument("--with-lora", action="store_true", default=True, help="Include LoRA point in trade-off plot (default: True)")
    parser.add_argument("--without-lora", dest="with_lora", action="store_false", help="Exclude LoRA point")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        script_root = Path(__file__).resolve().parents[1]
        candidate = script_root / args.outdir
        if candidate.exists() or not outdir.exists():
            outdir = candidate
        else:
            outdir = Path.cwd() / args.outdir

    _ensure_outdir(outdir)
    print(f"[info] Output directory: {outdir.resolve()}  dpi={args.dpi}  with_lora={args.with_lora}")

    plot_pareto_frontier(outdir, dpi=args.dpi, with_lora=args.with_lora)
    plot_structural_cascade(outdir, dpi=args.dpi)
    plot_leakage_and_confusion(outdir, dpi=args.dpi)

    print("[done] 3 figures generated successfully (Vector PDF + 300 DPI PNG)")

if __name__ == "__main__":
    main()
