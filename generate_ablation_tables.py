"""
generate_ablation_tables.py
----------------------------
Generates high-resolution, LaTeX-style ablation study table images for the
PhysX-MKS-GhostNet paper.  Produces three PNG figures:

  outputs/visualizations/Fig_Table8_Efficiency.png
  outputs/visualizations/Fig_Table7_EOC_Results.png
  outputs/visualizations/Fig_Tables56_SOC_Results.png

Each figure uses matplotlib's mathtext / rcParams to mimic a LaTeX-typeset
academic table with proper booktabs-style horizontal rules and a visually
distinct highlighted row for the proposed PhysX-MKS-GhostNet model.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(
    os.path.dirname(__file__), "outputs", "visualizations"
)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------
FONT_FAMILY   = "DejaVu Serif"   # closest to Computer Modern available
TITLE_SIZE    = 11
HEADER_SIZE   = 10
CELL_SIZE     = 9.5
HIGHLIGHT_CLR = "#d4e8ff"        # light blue tint for the proposed model row
RULE_CLR      = "#222222"
HEADER_BG     = "#f0f0f0"

plt.rcParams.update({
    "font.family":       FONT_FAMILY,
    "font.size":         CELL_SIZE,
    "axes.spines.left":  False,
    "axes.spines.right": False,
    "axes.spines.top":   False,
    "axes.spines.bottom":False,
})


# ---------------------------------------------------------------------------
# Helper: draw a single academic-style table on *ax*
# ---------------------------------------------------------------------------
def draw_table(ax, title, col_labels, rows, highlight_last=True,
               col_widths=None):
    """
    Render a booktabs-style table onto *ax*.

    Parameters
    ----------
    ax            : matplotlib Axes  (should have no visible spines / ticks)
    title         : str   — caption shown above the table
    col_labels    : list[str]
    rows          : list[list[str]]
    highlight_last: bool  — shade the last row to emphasise the proposed model
    col_widths    : list[float] or None  — relative column widths (sum = 1)
    """
    ax.set_axis_off()

    n_cols = len(col_labels)
    n_rows = len(rows)

    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols

    # Normalise to [0, 1]
    total = sum(col_widths)
    col_widths = [w / total for w in col_widths]

    # Compute cumulative x-positions for column left edges
    col_x = [0.0]
    for w in col_widths[:-1]:
        col_x.append(col_x[-1] + w)

    # Row height in axes-fraction units
    row_h    = 0.072
    title_h  = 0.10
    header_h = 0.082
    pad_top  = 0.04   # padding above title

    total_h = title_h + header_h + n_rows * row_h + pad_top
    # We'll work in [0, 1] y-space and scale
    y_scale = 1.0 / total_h if total_h > 0 else 1.0

    def y(row_idx):
        """Return the bottom y-coord of the given row (0 = header, 1..n = data)."""
        if row_idx == 0:      # header row
            return (total_h - pad_top - title_h - header_h) * y_scale
        else:
            return (total_h - pad_top - title_h - header_h - row_idx * row_h) * y_scale

    top_y = (total_h - pad_top) * y_scale

    # -- Title --
    title_y = (total_h - pad_top - title_h / 2) * y_scale
    ax.text(0.5, title_y, title,
            ha="center", va="center",
            fontsize=TITLE_SIZE, fontweight="bold",
            transform=ax.transAxes)

    # -- Top rule --
    rule_y_top = (total_h - pad_top - title_h) * y_scale
    ax.plot([0, 1], [rule_y_top, rule_y_top],
            color=RULE_CLR, linewidth=1.4,
            transform=ax.transAxes, clip_on=False)

    # -- Header background --
    hdr_bottom = y(0)
    hdr_rect = FancyBboxPatch(
        (0, hdr_bottom),
        1.0,
        header_h * y_scale,
        boxstyle="square,pad=0",
        facecolor=HEADER_BG,
        edgecolor="none",
        transform=ax.transAxes,
        clip_on=False,
        zorder=1,
    )
    ax.add_patch(hdr_rect)

    # -- Column headers --
    for ci, (label, cx, cw) in enumerate(zip(col_labels, col_x, col_widths)):
        ha = "left" if ci == 0 else "center"
        x_pos = cx + (0 if ci == 0 else cw / 2)
        ax.text(x_pos, hdr_bottom + (header_h * y_scale) / 2,
                label,
                ha=ha, va="center",
                fontsize=HEADER_SIZE, fontweight="bold",
                transform=ax.transAxes, zorder=2)

    # -- Mid rule (below header) --
    ax.plot([0, 1], [hdr_bottom, hdr_bottom],
            color=RULE_CLR, linewidth=0.9,
            transform=ax.transAxes, clip_on=False)

    # -- Data rows --
    for ri, row in enumerate(rows):
        row_bottom = y(ri + 1)
        is_last    = (ri == n_rows - 1) and highlight_last

        # Highlight last (proposed model) row
        if is_last:
            rect = FancyBboxPatch(
                (0, row_bottom),
                1.0,
                row_h * y_scale,
                boxstyle="square,pad=0",
                facecolor=HIGHLIGHT_CLR,
                edgecolor="none",
                transform=ax.transAxes,
                clip_on=False,
                zorder=1,
            )
            ax.add_patch(rect)

        for ci, (cell, cx, cw) in enumerate(zip(row, col_x, col_widths)):
            ha = "left" if ci == 0 else "center"
            x_pos = cx + (0 if ci == 0 else cw / 2)
            weight = "bold" if is_last else "normal"
            ax.text(x_pos,
                    row_bottom + (row_h * y_scale) / 2,
                    str(cell),
                    ha=ha, va="center",
                    fontsize=CELL_SIZE,
                    fontweight=weight,
                    transform=ax.transAxes, zorder=2)

    # -- Bottom rule --
    bottom_rule_y = y(n_rows)
    ax.plot([0, 1], [bottom_rule_y, bottom_rule_y],
            color=RULE_CLR, linewidth=1.4,
            transform=ax.transAxes, clip_on=False)

    # Fix y-limits so spacing is respected
    ax.set_ylim(bottom_rule_y - 0.02, top_y + 0.02)
    ax.set_xlim(0, 1)


# ===========================================================================
# Figure 1 — Table 8: Efficiency and Edge Deployment Metrics
# ===========================================================================
def make_table8():
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    fig.patch.set_facecolor("white")

    title = "Table 8: Efficiency and Edge Deployment Metrics: Pi 5 vs. Workstation"
    col_labels = ["Model", "Hardware", "Params", "GFLOPs", "Latency", "FPS"]
    col_widths = [2.8, 2.0, 1.3, 1.3, 1.5, 1.2]

    rows = [
        ["ResNet-18",              "Pi 5 (CPU)",    "47.8 MB", "1.27",  "~200 ms", "~5.0"],
        ["MobileNet-V2",           "Pi 5 (CPU)",    "14.0 MB", "0.30",  "~66.7 ms","~15.0"],
        ["CRMC-Net",               "RTX 3080 Ti",   "25.8 MB", "1.31",  "-",       "-"],
        ["PhysX-MKS-GhostNet",     "Pi 5 (CPU)",    "2.2 MB",  "4.2",   "61.7 ms", "16.4"],
    ]

    draw_table(ax, title, col_labels, rows,
               highlight_last=True, col_widths=col_widths)

    plt.tight_layout(pad=0.4)
    out = os.path.join(OUT_DIR, "Fig_Table8_Efficiency.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅  Saved {out}")


# ===========================================================================
# Figure 2 — Table 7: EOC Results
# ===========================================================================
def make_table7():
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    fig.patch.set_facecolor("white")

    title = "Table 7: EOC Results"
    col_labels = ["Model", "Type", "Loss", "Accuracy (%)"]
    col_widths = [3.2, 1.6, 1.4, 1.8]

    rows = [
        ["Traditional CNN",          "Real",    "0.3316", "91.33"],
        ["VGG16",                    "Real",    "0.3849", "88.68"],
        ["ResNet18",                 "Real",    "0.5477", "89.97"],
        ["CVCNN",                    "Complex", "0.2451", "93.09"],
        ["CV-Net",                   "Complex", "0.1727", "94.86"],
        ["CRMC-Net",                 "Complex", "0.1383", "95.02"],
        ["PhysX-MKS-Ghost (Ours)",   "PhysX",   "0.6026", "90.42"],
    ]

    draw_table(ax, title, col_labels, rows,
               highlight_last=True, col_widths=col_widths)

    plt.tight_layout(pad=0.4)
    out = os.path.join(OUT_DIR, "Fig_Table7_EOC_Results.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅  Saved {out}")


# ===========================================================================
# Figure 3 — Table 5 (SOC10) stacked above Table 6 (SOC3)
# ===========================================================================
def make_tables56():
    fig, (ax5, ax6) = plt.subplots(2, 1, figsize=(7.5, 6.2))
    fig.patch.set_facecolor("white")

    col_labels = ["Model", "Type", "Loss", "Accuracy (%)"]
    col_widths  = [3.2, 1.6, 1.4, 1.8]

    # -- Table 5: SOC10 --
    rows5 = [
        ["Traditional CNN",          "Real",    "0.3546", "96.25"],
        ["VGG16",                    "Real",    "0.0866", "97.65"],
        ["ResNet18",                 "Real",    "0.1469", "96.82"],
        ["CVCNN",                    "Complex", "0.0655", "98.59"],
        ["CV-Net",                   "Complex", "0.0225", "99.67"],
        ["CRMC-Net",                 "Complex", "0.0114", "99.83"],
        ["PhysX-MKS-Ghost (Ours)",   "PhysX",   "0.1569", "99.75"],
    ]
    draw_table(ax5, "Table 5: SOC10 Results", col_labels, rows5,
               highlight_last=True, col_widths=col_widths)

    # -- Table 6: SOC3 --
    rows6 = [
        ["Traditional CNN",          "Real",    "0.0273", "99.41"],
        ["VGG16",                    "Real",    "0.0315", "98.97"],
        ["ResNet18",                 "Real",    "0.0358", "99.05"],
        ["CVCNN",                    "Complex", "0.0329", "99.26"],
        ["CV-Net",                   "Complex", "0.0076", "99.83"],
        ["CRMC-Net",                 "Complex", "0.0029", "100.0"],
        ["PhysX-MKS-Ghost (Ours)",   "PhysX",   "0.1686", "99.41"],
    ]
    draw_table(ax6, "Table 6: SOC3 Results", col_labels, rows6,
               highlight_last=True, col_widths=col_widths)

    plt.tight_layout(pad=0.6, h_pad=1.8)
    out = os.path.join(OUT_DIR, "Fig_Tables56_SOC_Results.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅  Saved {out}")


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    print("Generating ablation study table figures ...")
    make_table8()
    make_table7()
    make_tables56()
    print("Done.  All figures written to:", OUT_DIR)
