"""Step 4: figures for the spatial-prompt section, in the paper's style (Fig. 5, 11a, 12).

    python plots.py
Run after metrics.py and calibrate.py. Writes PNG + PDF to outputs/figures/.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from common import ROOT

OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(OUT, "figures")

RED, BLUE, GREEN, PURPLE, BLACK = "#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#333333"
PERT_STYLE = {  # paper Fig. 11a legend order
    "shift": dict(color=RED, marker="o", label="Shift"),
    "downscale": dict(color=BLUE, marker="s", label="Downscale"),
    "upscale": dict(color=GREEN, marker="^", label="Upscale"),
}
MODEL_STYLE = {  # paper Fig. 12 legend order
    "DAM": dict(color=RED, marker="o", label="DAM"),
    "GPT4Roi": dict(color=BLUE, marker="s", label="GPT4Roi"),
    "Shikra": dict(color=GREEN, marker="*", label="Shikra", ms=8),
    "VipLlava": dict(color=PURPLE, marker="^", label="Vip_Llava"),
}
METRICS = [("OIP", True), ("CSR", False), ("APR", True), ("HAR", False)]  # (name, higher is better)
LEVELS = [0, 1, 3, 5]  # 0 = clean (exact) box
LEVEL_LABELS = ["Clean", "1", "3", "5"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.facecolor": "#f2f2f2", "axes.edgecolor": "white",
    "axes.grid": True, "grid.color": "white", "grid.linewidth": 1.0, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlesize": 11, "xtick.bottom": False, "ytick.left": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "legend.fancybox": False, "legend.shadow": True,
    "legend.edgecolor": "#999999", "legend.facecolor": "#f2f2f2",
})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)


def curve(summary, model, pert, metric):
    """Clean value followed by L1, L3, L5."""
    d = summary[summary.model == model]
    clean = d[d.perturbation == "clean"][metric].tolist()
    return clean + d[d.perturbation == pert].sort_values("level")[metric].tolist()


def series(ax, y, style, ls="-"):
    ax.plot(LEVELS[-len(y):], y, ls=ls, lw=2, ms=style.get("ms", 6), color=style["color"],
            marker=style["marker"], label=style["label"])


def fig_calibration():
    cal = os.path.join(OUT, "calibration")
    rand = np.load(os.path.join(cal, "random.npy"))
    matched = np.load(os.path.join(cal, "matched_obj.npy"))
    syn = pd.read_csv(os.path.join(cal, "synonyms.csv"))["cos"].to_numpy()
    floor = rand.mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("white")
    x0, x1, ytop = -0.15, 1.05, 5.4
    for lo, hi, color in [(x0, floor, "#e8e8e8"), (floor, 0.6, "#f9e0e0"), (0.6, 0.7, "#fde6c8"), (0.7, x1, "#dcefdc")]:
        ax.axvspan(lo, hi, color=color, zorder=0)
    for x, text, color in [((x0 + floor) / 2, "Random\nnoise", "#555555"), ((floor + 0.6) / 2, "Below\nthreshold", "#c0392b"),
                           (0.65, "Attribute\nmatch", "#d35400"), ((0.7 + x1) / 2, "Object identity\nmatch", "#1e7b1e")]:
        ax.text(x, 4.3, text, ha="center", va="center", color=color, fontsize=9, fontweight="bold", style="italic")

    # Cosine similarity cannot exceed 1. Pairs with identical phrases sit exactly at 1.0 (a point mass a
    # KDE cannot represent), so they are drawn as a separate bar; the KDE covers the remaining pairs, is
    # reflected at 1.0 so no density spills past it, and is scaled by their share.
    grid = np.linspace(x0, 1.0, 1500)
    rng = np.random.default_rng(0)
    exact = matched >= 0.9999
    peak_sigma = 0.03  # width of the drawn peak for the identical-name pairs
    for data, line, fill, label, is_matched in [
        (rand, "#444444", "#9e9e9e", f"Random object–attribute pairs (N={len(rand):,})", False),
        (matched[~exact], "#1f4fa8", "#6aa5dd", f"Reference–generated object pairs (N={len(matched):,})", True),
    ]:
        sample = data if len(data) <= 30000 else rng.choice(data, 30000, replace=False)
        kde = gaussian_kde(sample, bw_method=0.2)
        dens = kde(grid) + kde(2.0 - grid)
        if is_matched:
            # Weight the continuous part by its share and add the identical-name mass as a half-Gaussian at
            # 1.0 whose area equals that share, so the whole curve still integrates to 1.
            dens = dens * (~exact).mean() + exact.mean() * 2 * np.exp(-0.5 * ((grid - 1.0) / peak_sigma) ** 2) / (
                peak_sigma * np.sqrt(2 * np.pi))
        ax.plot(grid, dens, color=line, lw=2, label=label)
        ax.fill_between(grid, dens, color=fill, alpha=0.45)
        if is_matched:
            peak = dens[-1]

    ax.annotate(f"identical names\n(cos = 1): {exact.mean():.1%}", xy=(0.99, peak * 0.8), xytext=(0.86, peak * 0.62),
                ha="center", va="center", fontsize=8, color="#1f4fa8", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#1f4fa8", lw=1))

    ax.scatter(syn, rng.uniform(0.1, 0.3, len(syn)), s=60, color="#ff7f0e", edgecolor="white", lw=1, zorder=5,
               label=f"Curated synonym pairs (N={len(syn)})")
    ax.axvline(floor, color="#d62728", ls="--", lw=1.8, label=rf"$b_{{floor}}$ = {floor:.4f}")
    ax.axvline(0.6, color="#d62728", ls="--", lw=1.8, label="APR/HAR threshold = 0.6")
    ax.axvline(0.7, color="#1e7b1e", ls="--", lw=1.8, label="OIP threshold = 0.7")

    ax.set_xlim(x0, x1)
    ax.set_ylim(0, ytop)
    ax.set_xlabel("Cosine Similarity", fontweight="bold", fontsize=11)
    ax.set_ylabel("Density", fontweight="bold", fontsize=11)
    ax.set_title("Distribution of Cosine Similarities:\nRandom Noise Floor vs. Reference–Generated Object Pairs vs. Synonym Pairs",
                 fontsize=12)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")
    ax.tick_params(bottom=True, left=True)
    ax.grid(color="#dddddd", lw=0.6)
    # matplotlib fills legend columns first: (random, matched) | (synonyms, floor) | (0.6, 0.7), as in the paper
    by_label = dict(zip(*reversed(ax.get_legend_handles_labels())))
    keys = ["Random", "Reference", "Curated", "$b_", "APR/HAR", "OIP"]
    labels = [next(l for l in by_label if l.startswith(k)) for k in keys]
    ax.legend([by_label[l] for l in labels], labels, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, shadow=False, facecolor="white")
    save(fig, "fig5_calibration")


def fig_dam(summary):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, (m, higher) in zip(axes, METRICS):
        for pert, style in PERT_STYLE.items():
            series(ax, curve(summary, "DAM", pert, m), style)
        ax.set_title(m)
        ax.set_xticks(LEVELS, LEVEL_LABELS)
        ax.set_xlabel("Severity Level", fontsize=10)
        ax.set_ylabel(f"Score ({'Higher' if higher else 'Lower'} is better)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Noise Type", loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12),
               fontsize=10, title_fontsize=11)
    fig.tight_layout()
    save(fig, "fig11a_dam_spatial")


def fig_cross_model(summary):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    for row, pert in enumerate(["shift", "upscale"]):
        for col, (m, _) in enumerate(METRICS):
            ax = axes[row, col]
            curves = []
            for model, style in MODEL_STYLE.items():
                y = curve(summary, model, pert, m)[1:]  # Fig. 12 shows L1, L3, L5 only
                series(ax, y, style)
                curves.append(y)
            series(ax, np.mean(curves, axis=0), dict(color=BLACK, marker="D", label="Mean", ms=7), ls="--")
            if row == 0:
                ax.set_title(m)
            ax.set_xticks(LEVELS[1:], LEVEL_LABELS[1:])
            ax.set_xlabel("Severity Level", fontsize=10)
        axes[row, 0].set_ylabel(f"{pert.capitalize()} Score", fontweight="bold", fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.04), fontsize=10)
    fig.tight_layout(h_pad=3)
    save(fig, "fig12_cross_model")


def main():
    summary = pd.read_csv(os.path.join(OUT, "metrics_spatial.csv"))
    fig_calibration()
    fig_dam(summary)
    fig_cross_model(summary)
    print("saved to", FIG)


if __name__ == "__main__":
    main()
