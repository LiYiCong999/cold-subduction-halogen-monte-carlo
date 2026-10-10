# -*- coding: utf-8 -*-
"""
Style-matched version of the mélange Monte Carlo comparison figure.

Main purpose of this version:
    Keep the same scientific logic as the previous script, but make the final
    output figure style closer to the user-provided reference image, including:
    - font family
    - panel-title style
    - border and axis line thickness
    - tick style
    - histogram frame style
    - legend frame style
    - overall panel/figure appearance

Scientific layout:
    Rows:
        1) Blueschist-based mélange
        2) HP eclogite-based mélange
        3) UHP eclogite-based mélange

    Columns:
        left  = F/Cl vs I/Cl
        right = Br/Cl vs I/Cl

    In each panel, three mélange types are overlaid:
        - sediment-dominated
        - metamafic-dominated
        - serpentinite-dominated
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

# ============================================================
# USER SETTINGS
# ============================================================

INPUT_CSV = "data/halogen_data.csv"
OUTPUT_DIR = Path("results/NC_melange_MC_stylematch_v7_output")

N_ITER_PER_ROW = 300_000
RANDOM_SEED = 20261003

X_LIMITS = (1e-6, 1e-1)
Y_LIMITS_FCL = (1e-4, 1e2)
Y_LIMITS_BRCL = (1e-4, 1e-1)

# Marginal histograms: fewer, wider bins than the previous version.
# This is visually closer to the supplied reference figure and makes the
# three overlaid mélange distributions easier to distinguish.
HIST_BINS = 14

# One color family per row, with deliberately larger light–dark contrast.
# Colors are tuned to the reference figure:
#   Row 1 = cyan-blue
#   Row 2 = light brown-orange
#   Row 3 = muted gray-green
ROW_COLORSETS = {
    "Blueschist": {
        "metasediment-dominated": "#C4E2E8",   # light cyan-blue
        "metamafic-dominated":    "#6DB7C8",   # medium cyan-blue
        "serpentinite-dominated": "#237F99",   # dark cyan-blue
    },
    "HP eclogite": {
        "metasediment-dominated": "#F1D7BD",   # light sand-orange
        "metamafic-dominated":    "#E0A268",   # medium orange-brown
        "serpentinite-dominated": "#B96A2C",   # dark orange-brown
    },
    "UHP eclogite": {
        "metasediment-dominated": "#CDD9C5",   # light gray-green
        "metamafic-dominated":    "#8EAD79",   # medium gray-green
        "serpentinite-dominated": "#557D49",   # dark gray-green
    },
}


ROW_ORDER = ["Blueschist", "HP eclogite", "UHP eclogite"]
ROW_TITLES = {
    "Blueschist": "Blueschist domain (1)",
    "HP eclogite": "HP eclogite domain (2)",
    "UHP eclogite": "UHP eclogite domain (3)",
}
PANEL_LETTERS = ["A", "B", "C", "D", "E", "F"]

FIG_FACE = "#F3F3F3"
AX_FACE = "#F3F3F3"

# ============================================================
# STYLE
# ============================================================

def apply_reference_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10.5,
        "axes.linewidth": 0.8,
        "axes.facecolor": AX_FACE,
        "figure.facecolor": FIG_FACE,
        "savefig.facecolor": FIG_FACE,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "#555555",
        "legend.facecolor": "white",
    })


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_divide(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (b != 0)
    out[mask] = a[mask] / b[mask]
    return out


def summarize_array(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "p2.5": np.nan,
            "p16": np.nan,
            "p84": np.nan,
            "p97.5": np.nan,
        }
    return {
        "n": len(x),
        "mean": x.mean(),
        "median": x.median(),
        "p2.5": np.quantile(x, 0.025),
        "p16": np.quantile(x, 0.16),
        "p84": np.quantile(x, 0.84),
        "p97.5": np.quantile(x, 0.975),
    }


def summarize_result(df, group_cols, value_cols):
    rows = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = {group_cols[i]: keys[i] for i in range(len(group_cols))}
        for col in value_cols:
            rows.append({
                **base,
                "variable": col,
                **summarize_array(sub[col].to_numpy())
            })
    return pd.DataFrame(rows)


def draw_density_contours(ax, x, y, color, xlim, ylim):
    """
    Draw one single colored shadow for each mélange type.

    Important:
    - No internal density contour lines.
    - No light-to-dark density gradient.
    - One translucent filled envelope only.
    - One clear outer boundary in the same color.

    The envelope is defined by the 20% normalized 2-D density contour.
    This preserves the overall Monte Carlo distribution field while avoiding
    the appearance of internal density structure.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]

    if len(x) < 20:
        return

    lx = np.log10(x)
    ly = np.log10(y)

    xedges = np.linspace(np.log10(xlim[0]), np.log10(xlim[1]), 140)
    yedges = np.linspace(np.log10(ylim[0]), np.log10(ylim[1]), 140)

    H, xe, ye = np.histogram2d(lx, ly, bins=[xedges, yedges])
    Hs = gaussian_filter(H.T, sigma=1.15)

    if Hs.max() <= 0:
        return

    Hn = Hs / Hs.max()

    xc = 10 ** ((xe[:-1] + xe[1:]) / 2.0)
    yc = 10 ** ((ye[:-1] + ye[1:]) / 2.0)
    X, Y = np.meshgrid(xc, yc)

    # Single envelope threshold.
    shadow_level = 0.20

    # One uniform translucent shadow.
    ax.contourf(
        X,
        Y,
        Hn,
        levels=[shadow_level, 1.000001],
        colors=[color],
        alpha=0.30,
        antialiased=True,
        zorder=2,
    )

    # One clear outer boundary only.
    ax.contour(
        X,
        Y,
        Hn,
        levels=[shadow_level],
        colors=[color],
        linewidths=1.25,
        alpha=1.0,
        zorder=3,
    )

def draw_top_hist(ax, x, color, xlim):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) == 0:
        return

    lx = np.log10(x)
    bins = np.linspace(np.log10(xlim[0]), np.log10(xlim[1]), HIST_BINS + 1)
    counts, edges = np.histogram(lx, bins=bins)
    centers = 10 ** ((edges[:-1] + edges[1:]) / 2.0)
    widths = np.diff(10 ** edges)

    ax.bar(
        centers, counts, width=widths,
        color=color,
        alpha=0.56,
        edgecolor=color,
        linewidth=0.70,
        align="center",
        zorder=2,
    )

    try:
        grid = np.linspace(lx.min(), lx.max(), 400)
        kde = gaussian_kde(lx)
        dens = kde(grid)
        if dens.max() > 0 and counts.max() > 0:
            dens = dens / dens.max() * counts.max()
        ax.plot(
            10 ** grid, dens,
            color=color,
            lw=1.20,
            alpha=1.0,
            zorder=3,
        )
    except Exception:
        pass

    ax.set_xscale("log")
    ax.set_xlim(*xlim)


def draw_right_hist(ax, y, color, ylim):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y) & (y > 0)]
    if len(y) == 0:
        return

    ly = np.log10(y)
    bins = np.linspace(np.log10(ylim[0]), np.log10(ylim[1]), HIST_BINS + 1)
    counts, edges = np.histogram(ly, bins=bins)
    centers = 10 ** ((edges[:-1] + edges[1:]) / 2.0)
    heights = np.diff(10 ** edges)

    ax.barh(
        centers, counts, height=heights,
        color=color,
        alpha=0.56,
        edgecolor=color,
        linewidth=0.70,
        align="center",
        zorder=2,
    )

    try:
        grid = np.linspace(ly.min(), ly.max(), 400)
        kde = gaussian_kde(ly)
        dens = kde(grid)
        if dens.max() > 0 and counts.max() > 0:
            dens = dens / dens.max() * counts.max()
        ax.plot(
            dens, 10 ** grid,
            color=color,
            lw=1.20,
            alpha=1.0,
            zorder=3,
        )
    except Exception:
        pass

    ax.set_yscale("log")
    ax.set_ylim(*ylim)


def histogram_peak(values, limits):
    """Peak bin count using exactly the same logarithmic bins as the plotted histogram."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        return 0
    lv = np.log10(values)
    edges = np.linspace(np.log10(limits[0]), np.log10(limits[1]), HIST_BINS + 1)
    counts, _ = np.histogram(lv, bins=edges)
    return int(counts.max()) if len(counts) else 0


def nice_count_ceiling(value):
    """Round a count maximum upward to a clean 1/2/2.5/5/10 × 10^n value."""
    value = float(value)
    if value <= 0:
        return 1.0

    exponent = np.floor(np.log10(value))
    scale = 10.0 ** exponent
    fraction = value / scale

    if fraction <= 1:
        nice = 1.0
    elif fraction <= 2:
        nice = 2.0
    elif fraction <= 2.5:
        nice = 2.5
    elif fraction <= 5:
        nice = 5.0
    else:
        nice = 10.0

    return nice * scale


def compute_marginal_count_limits(df):
    """
    Use common count scales:
      - one common top-histogram scale for all six panels (I/Cl)
      - one common right-histogram scale for all F/Cl panels
      - one common right-histogram scale for all Br/Cl panels
    """
    top_peak = 0
    fcl_peak = 0
    brcl_peak = 0

    for metamafic_type in ROW_ORDER:
        row_df = df[df["metamafic_type"] == metamafic_type]

        for melange_name in [
            "metasediment-dominated",
            "metamafic-dominated",
            "serpentinite-dominated",
        ]:
            sub = row_df[row_df["melange_type"] == melange_name]

            top_peak = max(
                top_peak,
                histogram_peak(sub["I_Cl"].to_numpy(), X_LIMITS),
            )
            fcl_peak = max(
                fcl_peak,
                histogram_peak(sub["F_Cl"].to_numpy(), Y_LIMITS_FCL),
            )
            brcl_peak = max(
                brcl_peak,
                histogram_peak(sub["Br_Cl"].to_numpy(), Y_LIMITS_BRCL),
            )

    # Add a small headroom before rounding upward.
    return (
        nice_count_ceiling(top_peak * 1.08),
        nice_count_ceiling(fcl_peak * 1.08),
        nice_count_ceiling(brcl_peak * 1.08),
    )

def format_main_axis(ax, xlabel, ylabel, xlim, ylim):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel, fontsize=11, fontweight="normal")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="normal")
    ax.tick_params(which="both", length=4, width=0.8, labelsize=9.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _format_count_tick(value, pos):
    if abs(value) < 1e-12:
        return "0"
    return f"{int(round(value))}"


def format_top_axis(ax, xlim, count_max):
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(0, count_max)
    ax.set_yticks([0, count_max])
    ax.yaxis.set_major_formatter(FuncFormatter(_format_count_tick))

    ax.tick_params(
        axis="x",
        labelbottom=False,
        which="both",
        length=4,
        width=0.8,
    )
    ax.tick_params(
        axis="y",
        labelsize=8.2,
        length=3.2,
        width=0.75,
        pad=1.5,
    )

    ax.yaxis.tick_right()
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    for s in ["top", "right"]:
        ax.spines[s].set_linewidth(0.8)


def format_right_axis(ax, ylim, count_max):
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_xlim(0, count_max)
    ax.set_xticks([0, count_max])
    ax.xaxis.set_major_formatter(FuncFormatter(_format_count_tick))

    ax.tick_params(
        axis="y",
        labelleft=False,
        which="both",
        length=4,
        width=0.8,
    )
    ax.tick_params(
        axis="x",
        labelsize=8.2,
        length=3.2,
        width=0.75,
        pad=1.5,
    )

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # The supplied reference figure uses the count values themselves rather
    # than repeating a "Count" label in every marginal panel.
    ax.set_xlabel("")

    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    for s in ["top", "right"]:
        ax.spines[s].set_linewidth(0.8)



# ============================================================
# DATA
# ============================================================

def load_data(path):
    df = pd.read_csv(path)

    required_cols = ["Rock_Type", "F_ppm", "Cl_ppm", "Br_ppb", "I_ppb"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in ["F_ppm", "Cl_ppm", "Br_ppb", "I_ppb"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Rock_Type", "F_ppm", "Cl_ppm", "Br_ppb", "I_ppb"]).copy()
    df = df[(df["F_ppm"] >= 0) & (df["Cl_ppm"] >= 0) &
            (df["Br_ppb"] >= 0) & (df["I_ppb"] >= 0)].copy()

    used_types = {"metasediment", "forearc serpentinite", "Blueschist", "HP eclogite", "UHP eclogite"}
    df = df[df["Rock_Type"].isin(used_types)].copy()

    print("Usable sample counts by Rock_Type:")
    print(df["Rock_Type"].value_counts())

    return df


# ============================================================
# MONTE CARLO
# ============================================================

def run_one_row(df_all, metamafic_type, n_iter, seed):
    rng = np.random.default_rng(seed)

    sed = df_all[df_all["Rock_Type"] == "metasediment"].reset_index(drop=True)
    maf = df_all[df_all["Rock_Type"] == metamafic_type].reset_index(drop=True)
    serp = df_all[df_all["Rock_Type"] == "forearc serpentinite"].reset_index(drop=True)

    P = rng.dirichlet([1.0, 1.0, 1.0], size=n_iter)

    dominant_idx = np.argmax(P, axis=1)
    dominant_map = {
        0: "metasediment-dominated",
        1: "metamafic-dominated",
        2: "serpentinite-dominated",
    }
    melange_type = np.array([dominant_map[i] for i in dominant_idx])

    sed_s = sed.iloc[rng.integers(0, len(sed), size=n_iter)].reset_index(drop=True)
    maf_s = maf.iloc[rng.integers(0, len(maf), size=n_iter)].reset_index(drop=True)
    serp_s = serp.iloc[rng.integers(0, len(serp), size=n_iter)].reset_index(drop=True)

    F_mix = (
        P[:, 0] * sed_s["F_ppm"].to_numpy()
        + P[:, 1] * maf_s["F_ppm"].to_numpy()
        + P[:, 2] * serp_s["F_ppm"].to_numpy()
    )
    Cl_mix = (
        P[:, 0] * sed_s["Cl_ppm"].to_numpy()
        + P[:, 1] * maf_s["Cl_ppm"].to_numpy()
        + P[:, 2] * serp_s["Cl_ppm"].to_numpy()
    )
    Br_mix_ppb = (
        P[:, 0] * sed_s["Br_ppb"].to_numpy()
        + P[:, 1] * maf_s["Br_ppb"].to_numpy()
        + P[:, 2] * serp_s["Br_ppb"].to_numpy()
    )
    I_mix_ppb = (
        P[:, 0] * sed_s["I_ppb"].to_numpy()
        + P[:, 1] * maf_s["I_ppb"].to_numpy()
        + P[:, 2] * serp_s["I_ppb"].to_numpy()
    )

    Br_mix_ppm = Br_mix_ppb / 1000.0
    I_mix_ppm = I_mix_ppb / 1000.0

    out = pd.DataFrame({
        "metamafic_type": metamafic_type,
        "melange_type": melange_type,
        "metasediment_fraction": P[:, 0],
        "metamafic_fraction": P[:, 1],
        "serpentinite_fraction": P[:, 2],
        "F_ppm": F_mix,
        "Cl_ppm": Cl_mix,
        "Br_ppb": Br_mix_ppb,
        "I_ppb": I_mix_ppb,
        "F_Cl": safe_divide(F_mix, Cl_mix),
        "Br_Cl": safe_divide(Br_mix_ppm, Cl_mix),
        "I_Cl": safe_divide(I_mix_ppm, Cl_mix),
        "sum_fraction": P[:, 0] + P[:, 1] + P[:, 2],
    })
    return out


def run_all_rows(df, n_iter, seed):
    outputs = []
    for i, metamafic_type in enumerate(ROW_ORDER):
        outputs.append(run_one_row(df, metamafic_type, n_iter, seed + i * 100))
    return pd.concat(outputs, ignore_index=True)


# ============================================================
# FIGURE
# ============================================================

def build_figure(df, output_dir):
    apply_reference_style()

    fig = plt.figure(figsize=(13.6, 17.6), facecolor=FIG_FACE)
    outer = fig.add_gridspec(
        3, 2,
        left=0.065, right=0.975, top=0.985, bottom=0.05,
        wspace=0.205, hspace=0.19
    )

    # Common marginal-axis count scales make the six panels directly comparable.
    TOP_COUNT_MAX, RIGHT_COUNT_MAX_FCL, RIGHT_COUNT_MAX_BRCL = compute_marginal_count_limits(df)

    panel_idx = 0

    for row, metamafic_type in enumerate(ROW_ORDER):
        row_df = df[df["metamafic_type"] == metamafic_type].copy()
        row_colors = ROW_COLORSETS[metamafic_type]

        for col, (ycol, ylabel, ylim) in enumerate([
            ("F_Cl", "F/Cl", Y_LIMITS_FCL),
            ("Br_Cl", "Br/Cl", Y_LIMITS_BRCL),
        ]):
            subgrid = outer[row, col].subgridspec(
                2, 2,
                height_ratios=[0.82, 7.00],
                width_ratios=[7.00, 0.92],
                hspace=0.04,
                wspace=0.03,
            )

            ax_top = fig.add_subplot(subgrid[0, 0], facecolor=AX_FACE)
            ax_main = fig.add_subplot(subgrid[1, 0], facecolor=AX_FACE)
            ax_right = fig.add_subplot(subgrid[1, 1], sharey=ax_main, facecolor=AX_FACE)
            ax_blank = fig.add_subplot(subgrid[0, 1], facecolor=AX_FACE)
            ax_blank.axis("off")

            for melange_name in ["metasediment-dominated", "metamafic-dominated", "serpentinite-dominated"]:
                color = row_colors[melange_name]
                sub = row_df[row_df["melange_type"] == melange_name]

                x = sub["I_Cl"].to_numpy()
                y = sub[ycol].to_numpy()
                mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                x = x[mask]
                y = y[mask]

                draw_density_contours(ax_main, x, y, color, X_LIMITS, ylim)
                draw_top_hist(ax_top, x, color, X_LIMITS)
                draw_right_hist(ax_right, y, color, ylim)

            format_main_axis(ax_main, "I/Cl", ylabel, X_LIMITS, ylim)

            # Reference panel is a slightly wider-than-tall rectangle.
            # height / width ~= 0.875.
            ax_main.set_box_aspect(0.875)

            format_top_axis(ax_top, X_LIMITS, TOP_COUNT_MAX)

            if ycol == "F_Cl":
                right_count_max = RIGHT_COUNT_MAX_FCL
            else:
                right_count_max = RIGHT_COUNT_MAX_BRCL

            format_right_axis(ax_right, ylim, right_count_max)

            ax_main.text(
                0.06, 0.93, PANEL_LETTERS[panel_idx],
                transform=ax_main.transAxes,
                fontsize=13.5, fontweight="bold", ha="left", va="center"
            )

            ax_main.text(
                0.5, 0.95, ROW_TITLES[metamafic_type],
                transform=ax_main.transAxes,
                ha="center", va="center",
                fontsize=12.8, fontweight="normal"
            )

            handles = [
                Line2D([0], [0], color=row_colors["metasediment-dominated"],
                       lw=1.7, label="Sediment-dominated"),
                Line2D([0], [0], color=row_colors["metamafic-dominated"],
                       lw=1.7, label="Metamafic-dominated"),
                Line2D([0], [0], color=row_colors["serpentinite-dominated"],
                       lw=1.7, label="Serpentinite-dominated"),
            ]

            if col == 0:
                legend = ax_main.legend(
                    handles=handles,
                    loc="lower right",
                    fontsize=8.3,
                    handlelength=2.15,
                    borderpad=0.28,
                    labelspacing=0.25,
                    handletextpad=0.55,
                    frameon=True,
                    fancybox=False,
                    framealpha=1.0,
                    borderaxespad=0.65,
                )
                legend.get_frame().set_linewidth(0.75)
                legend.get_frame().set_edgecolor("#555555")
                legend.get_frame().set_facecolor("white")

            panel_idx += 1

    fig.savefig(output_dir / "Figure_compare_three_melanges_by_mafic_type_stylematch_v7.png", dpi=600)
    fig.savefig(output_dir / "Figure_compare_three_melanges_by_mafic_type_stylematch_v7.pdf")
    fig.savefig(output_dir / "Figure_compare_three_melanges_by_mafic_type_stylematch_v7.svg")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(INPUT_CSV)
    df.to_csv(OUTPUT_DIR / "input_data_used.csv", index=False)

    result = run_all_rows(df, N_ITER_PER_ROW, RANDOM_SEED)
    result.to_csv(OUTPUT_DIR / "melange_all_iterations_by_mafic_type.csv", index=False)

    summary = summarize_result(
        result,
        group_cols=["metamafic_type", "melange_type"],
        value_cols=[
            "metasediment_fraction", "metamafic_fraction", "serpentinite_fraction",
            "F_ppm", "Cl_ppm", "Br_ppb", "I_ppb", "F_Cl", "Br_Cl", "I_Cl"
        ]
    )
    summary.to_csv(OUTPUT_DIR / "melange_summary_by_mafic_type.csv", index=False)

    build_figure(result, OUTPUT_DIR)

    print("\nFinished successfully.")
    print(f"All outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
