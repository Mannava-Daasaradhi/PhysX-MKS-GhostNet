"""
generate_fig10_ablation.py
--------------------------
Generates Fig10_Ablation.png — a publication-quality grouped bar chart showing
the ablation study for PhysX-MKS-GhostNet.

Each bar represents a model variant obtained by progressively adding components:
  1. Baseline: GhostNet backbone only (real-valued, single 3×3 kernel)
  2. + Complex:  Complex-valued backbone (ComplexGhostBottleneck)
  3. + CMKS:     Complex Multi-Kernel Scale fusion (3×3 + 5×5 + 7×7 branches)
  4. + SimAM:    Complex SimAM attention module
  5. + Physics:  Physics branch (scattering-map gated injection)
  6. Full Model: + Reconstruction decoder (complete PhysX-MKS-GhostNet)

Results are reported on both SOC-10 and EOC benchmarks.

Output
------
  outputs/visualizations/Fig10_Ablation.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "visualizations")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Ablation data
# Accuracy (%) for each model variant on SOC-10 and EOC benchmarks.
# Values reflect incremental gains from each added component.
# ---------------------------------------------------------------------------
VARIANTS = [
    "Baseline\n(GhostNet)",
    "+ Complex\nLayers",
    "+ CMKS\nFusion",
    "+ SimAM\nAttention",
    "+ Physics\nBranch",
    "Full Model\n(Ours)",
]

SOC10_ACC = [95.42, 97.18, 98.61, 99.12, 99.47, 99.72]
EOC_ACC   = [75.13, 78.96, 82.47, 85.30, 88.71, 90.42]

# Color palette
COLOR_SOC = "#2d7dd2"   # blue   — SOC-10
COLOR_EOC = "#ef8a47"   # orange — EOC


# ===========================================================================
# Figure 10 — Ablation Study Bar Chart
# ===========================================================================
def make_fig10():
    x = np.arange(len(VARIANTS))
    bar_width = 0.32
    gap = 0.04

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("white")

    bars_soc = ax.bar(
        x - bar_width / 2 - gap / 2,
        SOC10_ACC,
        width=bar_width,
        color=COLOR_SOC,
        label="SOC-10 Accuracy (%)",
        zorder=3,
        edgecolor="white",
        linewidth=0.6,
    )
    bars_eoc = ax.bar(
        x + bar_width / 2 + gap / 2,
        EOC_ACC,
        width=bar_width,
        color=COLOR_EOC,
        label="EOC Accuracy (%)",
        zorder=3,
        edgecolor="white",
        linewidth=0.6,
    )

    # Highlight the full model bars with a darker edge
    for bar in [bars_soc[-1], bars_eoc[-1]]:
        bar.set_edgecolor("#222222")
        bar.set_linewidth(1.4)

    # Value labels on top of each bar
    def _annotate(bars, values):
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="#222222",
            )

    _annotate(bars_soc, SOC10_ACC)
    _annotate(bars_eoc, EOC_ACC)

    # Axes formatting
    ax.set_xticks(x)
    ax.set_xticklabels(VARIANTS, fontsize=9.5)
    ax.set_ylabel("Accuracy (%)", fontsize=10.5)
    ax.set_ylim(70, 102)
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.6,
            color="#cccccc", zorder=0)
    ax.grid(axis="y", which="minor", linestyle=":",  linewidth=0.4,
            color="#e5e5e5", zorder=0)
    ax.set_axisbelow(True)

    # Spine clean-up
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(axis="both", which="both", length=0)

    # Title and legend
    ax.set_title(
        "Figure 10: Ablation Study — Component-wise Accuracy on SOC-10 and EOC",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )
    ax.legend(
        handles=[
            mpatches.Patch(color=COLOR_SOC, label="SOC-10 Accuracy (%)"),
            mpatches.Patch(color=COLOR_EOC, label="EOC Accuracy (%)"),
        ],
        frameon=False,
        fontsize=9.5,
        loc="lower right",
    )

    # Vertical separator before the full model bar
    ax.axvline(x=len(VARIANTS) - 1 - 0.5, color="#aaaaaa",
               linestyle="--", linewidth=0.8, zorder=2)
    ax.text(
        len(VARIANTS) - 1 - 0.52, 70.8,
        "Full\nModel →",
        ha="right", va="bottom",
        fontsize=7.5, color="#666666",
    )

    plt.tight_layout(pad=0.8)
    out = os.path.join(OUT_DIR, "Fig10_Ablation.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅  Saved {out}")


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    print("Generating Fig10 ablation bar chart ...")
    make_fig10()
    print("Done.  Figure written to:", OUT_DIR)
