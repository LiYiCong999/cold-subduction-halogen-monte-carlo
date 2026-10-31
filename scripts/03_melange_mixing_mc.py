# -*- coding: utf-8 -*-
"""
Style-matched version of the mélange Monte Carlo comparison figure.

Modified version:
    - Monte Carlo iterations reduced to 10,000 per metamafic row
    - final output contains two figures only
    - both forearc- and subarc-serpentinite endmembers are simulated independently
    - both figures remove marginal histograms and distribution curves
    - Figure 1 shows 95% confidence/probability regions for the three dominant mélange types
    - Figure 2 shows the same results as scatter plots only


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

INPUT_CSV = Path(__file__).resolve().parents[1] / "data" / "halogen_data.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "NC_melange_MC_forearc_subarc_two_figures"

N_ITER_PER_ROW = 10_000
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

SERPENTINITE_ORDER = ["forearc serpentinite", "subarc serpentinite"]
SERPENTINITE_LABELS = {
    "forearc serpentinite": "Forearc serpentinite endmember",
    "subarc serpentinite": "Subarc serpentinite endmember",
}
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


def draw_ci95_region(ax, x, y, color, xlim, ylim):
    """
    Draw a 95% highest-density / confidence-like region in log-log space.

    For Monte Carlo results, this is implemented as a 2-D KDE on log10(I/Cl) and
    log10(y), then the density threshold is chosen so that the enclosed grid cells
    contain ~95% of the total KDE probability mass.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]

    if len(x) < 30:
        return

    lx = np.log10(x)
    ly = np.log10(y)

    try:
        kde = gaussian_kde(np.vstack([lx, ly]))
    except Exception:
        return

    gx = np.linspace(np.log10(xlim[0]), np.log10(xlim[1]), 180)
    gy = np.linspace(np.log10(ylim[0]), np.log10(ylim[1]), 180)
    Xg, Yg = np.meshgrid(gx, gy)
    positions = np.vstack([Xg.ravel(), Yg.ravel()])
    Z = kde(positions).reshape(Xg.shape)

    if not np.isfinite(Z).any() or np.nanmax(Z) <= 0:
        return

    flat = Z.ravel()
    order = np.argsort(flat)[::-1]
    flat_sorted = flat[order]
    csum = np.cumsum(flat_sorted)
    csum = csum / csum[-1]
    idx = np.searchsorted(csum, 0.95, side='left')
    idx = min(idx, len(flat_sorted)-1)
    level = flat_sorted[idx]

    X = 10 ** Xg
    Y = 10 ** Yg

    ax.contourf(
        X, Y, Z,
        levels=[level, Z.max() * (1.0 + 1e-10)],
        colors=[color],
        alpha=0.26,
        antialiased=True,
        zorder=2,
    )
    ax.contour(
        X, Y, Z,
        levels=[level],
        colors=[color],
        linewidths=1.25,
        alpha=1.0,
        zorder=3,
    )


def draw_square_scatter(ax, x, y, color):
    """Plot square scatter points as one coherent point cloud."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return

    ax.scatter(
        x, y,
        s=7.0,
        marker='s',
        facecolors=color,
        edgecolors='none',
        alpha=0.58,
        zorder=3,
        rasterized=True,
    )


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

def _canonical_rock_type(value):
    """Normalize common Rock_Type spellings to the canonical labels used below."""
    s = str(value).strip()
    key = " ".join(s.lower().replace("_", " ").replace("-", " ").split())

    mapping = {
        "metasediment": "metasediment",
        "metasediments": "metasediment",
        "forearc serpentinite": "forearc serpentinite",
        "forearc serpentinites": "forearc serpentinite",
        "fore arc serpentinite": "forearc serpentinite",
        "fore arc serpentinites": "forearc serpentinite",
        "subarc serpentinite": "subarc serpentinite",
        "subarc serpentinites": "subarc serpentinite",
        "sub arc serpentinite": "subarc serpentinite",
        "sub arc serpentinites": "subarc serpentinite",
        "blueschist": "Blueschist",
        "blueschists": "Blueschist",
        "hp eclogite": "HP eclogite",
        "hp eclogites": "HP eclogite",
        "high pressure eclogite": "HP eclogite",
        "high pressure eclogites": "HP eclogite",
        "uhp eclogite": "UHP eclogite",
        "uhp eclogites": "UHP eclogite",
        "ultrahigh pressure eclogite": "UHP eclogite",
        "ultrahigh pressure eclogites": "UHP eclogite",
    }
    return mapping.get(key, s)


def load_data(path):
    df = pd.read_csv(path)

    required_cols = ["Rock_Type", "F_ppm", "Cl_ppm", "Br_ppb", "I_ppb"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Rock_Type"] = df["Rock_Type"].map(_canonical_rock_type)

    for col in ["F_ppm", "Cl_ppm", "Br_ppb", "I_ppb"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Rock_Type", "F_ppm", "Cl_ppm", "Br_ppb", "I_ppb"]).copy()
    df = df[(df["F_ppm"] >= 0) & (df["Cl_ppm"] >= 0) &
            (df["Br_ppb"] >= 0) & (df["I_ppb"] >= 0)].copy()

    used_types = {
        "metasediment",
        "forearc serpentinite",
        "subarc serpentinite",
        "Blueschist",
        "HP eclogite",
        "UHP eclogite",
    }
    df = df[df["Rock_Type"].isin(used_types)].copy()

    required_types = [
        "metasediment",
        "forearc serpentinite",
        "subarc serpentinite",
        "Blueschist",
        "HP eclogite",
        "UHP eclogite",
    ]
    missing_types = [t for t in required_types if (df["Rock_Type"] == t).sum() == 0]
    if missing_types:
        raise ValueError(
            "No usable samples found for required Rock_Type(s): "
            + ", ".join(missing_types)
            + ".\nFor the new serpentinite scenario, accepted labels include "
              "'subarc serpentinite' or 'subarc serpentinites'."
        )

    print("Usable sample counts by Rock_Type:")
    print(df["Rock_Type"].value_counts())

    return df


# ============================================================
# MONTE CARLO
# ============================================================

def run_one_row(df_all, metamafic_type, serpentinite_type, n_iter, seed):
    """Run one metamafic stage using the selected serpentinite endmember."""
    rng = np.random.default_rng(seed)

    sed = df_all[df_all["Rock_Type"] == "metasediment"].reset_index(drop=True)
    maf = df_all[df_all["Rock_Type"] == metamafic_type].reset_index(drop=True)
    serp = df_all[df_all["Rock_Type"] == serpentinite_type].reset_index(drop=True)

    if len(serp) == 0:
        raise ValueError(f"No samples available for serpentinite endmember: {serpentinite_type}")

    # Same Dirichlet(1,1,1) mixing logic as before.
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
        "serpentinite_endmember": serpentinite_type,
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
    """Run both forearc- and subarc-serpentinite mixing scenarios."""
    outputs = []
    for serpentinite_type in SERPENTINITE_ORDER:
        for i, metamafic_type in enumerate(ROW_ORDER):
            # Use the same stage-specific seed for both serpentinite scenarios.
            # This keeps the mixing-fraction and metasediment/metamafic sampling
            # structure directly comparable; only the serpentinite endmember changes.
            outputs.append(
                run_one_row(
                    df,
                    metamafic_type=metamafic_type,
                    serpentinite_type=serpentinite_type,
                    n_iter=n_iter,
                    seed=seed + i * 100,
                )
            )
    return pd.concat(outputs, ignore_index=True)


# ============================================================
# FIGURE
# ============================================================

def _style_simple_panel(ax, ylabel, ylim, letter=None, title=None):
    format_main_axis(ax, "I/Cl", ylabel, X_LIMITS, ylim)
    ax.set_box_aspect(0.875)
    if letter is not None:
        ax.text(
            0.055, 0.93, letter,
            transform=ax.transAxes,
            fontsize=13.5, fontweight="bold", ha="left", va="center"
        )
    if title:
        ax.set_title(title, fontsize=12.3, fontweight="normal", pad=10)


def _add_hp_uhp_legend(ax):
    handles = []
    for metamafic_type, prefix in [("HP eclogite", "HP"), ("UHP eclogite", "UHP")]:
        colors = ROW_COLORSETS[metamafic_type]
        for melange_name, pretty in [
            ("metasediment-dominated", "Sediment-dominated"),
            ("metamafic-dominated", "Metamafic-dominated"),
            ("serpentinite-dominated", "Serpentinite-dominated"),
        ]:
            handles.append(
                Line2D([0], [0], color=colors[melange_name], lw=1.7, label=f"{prefix} {pretty}")
            )
    legend = ax.legend(
        handles=handles, loc="lower right", fontsize=7.5,
        handlelength=1.9, borderpad=0.28, labelspacing=0.18,
        handletextpad=0.45, frameon=True, fancybox=False,
        framealpha=1.0, borderaxespad=0.6, ncol=1,
    )
    legend.get_frame().set_linewidth(0.75)
    legend.get_frame().set_edgecolor("#555555")
    legend.get_frame().set_facecolor("white")


def build_figure_hp_uhp_by_serpentinite(df, output_dir):
    """
    Figure 1: HP + UHP 95% regions, 2 rows × 2 columns.

    Rows:
      1) F/Cl vs I/Cl
      2) Br/Cl vs I/Cl
    Columns:
      1) forearc serpentinite endmember
      2) subarc serpentinite endmember
    """
    apply_reference_style()

    fig = plt.figure(figsize=(13.8, 11.5), facecolor=FIG_FACE)
    outer = fig.add_gridspec(
        2, 2,
        left=0.075, right=0.975, top=0.945, bottom=0.075,
        wspace=0.20, hspace=0.24
    )

    row_specs = [
        ("F_Cl", "F/Cl", Y_LIMITS_FCL),
        ("Br_Cl", "Br/Cl", Y_LIMITS_BRCL),
    ]
    panel_letters = ["A", "B", "C", "D"]
    idx = 0

    for row, (ycol, ylabel, ylim) in enumerate(row_specs):
        for col, serpentinite_type in enumerate(SERPENTINITE_ORDER):
            ax = fig.add_subplot(outer[row, col], facecolor=AX_FACE)
            scenario_df = df[df["serpentinite_endmember"] == serpentinite_type]

            for metamafic_type in ["HP eclogite", "UHP eclogite"]:
                stage_df = scenario_df[scenario_df["metamafic_type"] == metamafic_type]
                stage_colors = ROW_COLORSETS[metamafic_type]

                for melange_name in [
                    "metasediment-dominated",
                    "metamafic-dominated",
                    "serpentinite-dominated",
                ]:
                    sub = stage_df[stage_df["melange_type"] == melange_name]
                    draw_ci95_region(
                        ax,
                        sub["I_Cl"].to_numpy(),
                        sub[ycol].to_numpy(),
                        stage_colors[melange_name],
                        X_LIMITS,
                        ylim,
                    )

            _style_simple_panel(
                ax, ylabel, ylim,
                letter=panel_letters[idx],
                title=SERPENTINITE_LABELS[serpentinite_type],
            )
            if row == 0 and col == 0:
                _add_hp_uhp_legend(ax)
            idx += 1

    fig.text(
        0.5, 0.985,
        "HP and UHP eclogite mélange: effect of serpentinite endmember",
        ha="center", va="top", fontsize=14.0, fontweight="normal"
    )

    fig.savefig(output_dir / "Figure1_HP_UHP_CI95_forearc_vs_subarc.png", dpi=600)
    fig.savefig(output_dir / "Figure1_HP_UHP_CI95_forearc_vs_subarc.pdf")
    fig.savefig(output_dir / "Figure1_HP_UHP_CI95_forearc_vs_subarc.svg")
    plt.close(fig)


def _add_three_melange_legend(ax, row_colors):
    handles = [
        Line2D([0], [0], color=row_colors["metasediment-dominated"], lw=1.7, label="Sediment-dominated"),
        Line2D([0], [0], color=row_colors["metamafic-dominated"], lw=1.7, label="Metamafic-dominated"),
        Line2D([0], [0], color=row_colors["serpentinite-dominated"], lw=1.7, label="Serpentinite-dominated"),
    ]
    legend = ax.legend(
        handles=handles, loc="lower right", fontsize=7.7,
        handlelength=2.0, borderpad=0.25, labelspacing=0.20,
        handletextpad=0.50, frameon=True, fancybox=False,
        framealpha=1.0, borderaxespad=0.55,
    )
    legend.get_frame().set_linewidth(0.75)
    legend.get_frame().set_edgecolor("#555555")
    legend.get_frame().set_facecolor("white")


def build_figure_all_12_panels(df, output_dir):
    """
    Figure 2: 12 panels.

    The original 2×3 layout is retained for each serpentinite endmember:
      Forearc block (rows 1–2): F/Cl, then Br/Cl; columns BS, HP, UHP.
      Subarc block  (rows 3–4): F/Cl, then Br/Cl; columns BS, HP, UHP.

    Each panel keeps the 95% region + rasterized scatter overlay.
    """
    apply_reference_style()

    fig = plt.figure(figsize=(17.2, 22.2), facecolor=FIG_FACE)
    outer = fig.add_gridspec(
        4, 3,
        left=0.055, right=0.985, top=0.955, bottom=0.045,
        wspace=0.20, hspace=0.26
    )

    stage_order = ["Blueschist", "HP eclogite", "UHP eclogite"]
    row_specs = [
        ("F_Cl", "F/Cl", Y_LIMITS_FCL),
        ("Br_Cl", "Br/Cl", Y_LIMITS_BRCL),
    ]
    panel_letters = list("ABCDEFGHIJKL")
    idx = 0

    for scenario_idx, serpentinite_type in enumerate(SERPENTINITE_ORDER):
        scenario_df = df[df["serpentinite_endmember"] == serpentinite_type]
        row_offset = scenario_idx * 2

        for local_row, (ycol, ylabel, ylim) in enumerate(row_specs):
            grid_row = row_offset + local_row

            for col, metamafic_type in enumerate(stage_order):
                ax = fig.add_subplot(outer[grid_row, col], facecolor=AX_FACE)
                stage_df = scenario_df[scenario_df["metamafic_type"] == metamafic_type]
                row_colors = ROW_COLORSETS[metamafic_type]

                for melange_name in [
                    "metasediment-dominated",
                    "metamafic-dominated",
                    "serpentinite-dominated",
                ]:
                    sub = stage_df[stage_df["melange_type"] == melange_name]
                    x = sub["I_Cl"].to_numpy()
                    y = sub[ycol].to_numpy()
                    color = row_colors[melange_name]
                    draw_ci95_region(ax, x, y, color, X_LIMITS, ylim)
                    draw_square_scatter(ax, x, y, color)

                format_main_axis(ax, "I/Cl", ylabel, X_LIMITS, ylim)
                ax.set_box_aspect(0.875)
                ax.text(
                    0.055, 0.93, panel_letters[idx],
                    transform=ax.transAxes, fontsize=13.0, fontweight="bold",
                    ha="left", va="center"
                )
                ax.text(
                    0.50, 0.95, ROW_TITLES[metamafic_type],
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11.8, fontweight="normal"
                )

                if local_row == 0 and col == 0:
                    _add_three_melange_legend(ax, row_colors)

                idx += 1

    # Block labels clearly distinguish the serpentinite endmember used.
    fig.text(
        0.52, 0.982, "Serpentinite endmember: Forearc serpentinite",
        ha="center", va="top", fontsize=14.0, fontweight="bold"
    )
    fig.text(
        0.52, 0.507, "Serpentinite endmember: Subarc serpentinite",
        ha="center", va="center", fontsize=14.0, fontweight="bold"
    )

    fig.savefig(output_dir / "Figure2_all12_forearc_subarc_CI95_plus_scatter.png", dpi=600)
    fig.savefig(output_dir / "Figure2_all12_forearc_subarc_CI95_plus_scatter.pdf")
    fig.savefig(output_dir / "Figure2_all12_forearc_subarc_CI95_plus_scatter.svg")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(INPUT_CSV)
    df.to_csv(OUTPUT_DIR / "input_data_used.csv", index=False)

    result = run_all_rows(df, N_ITER_PER_ROW, RANDOM_SEED)
    result.to_csv(OUTPUT_DIR / "melange_all_iterations_forearc_subarc.csv", index=False)

    summary = summarize_result(
        result,
        group_cols=["serpentinite_endmember", "metamafic_type", "melange_type"],
        value_cols=[
            "metasediment_fraction", "metamafic_fraction", "serpentinite_fraction",
            "F_ppm", "Cl_ppm", "Br_ppb", "I_ppb", "F_Cl", "Br_Cl", "I_Cl"
        ]
    )
    summary.to_csv(OUTPUT_DIR / "melange_summary_forearc_subarc.csv", index=False)

    # Final output: only the two requested figures.
    build_figure_hp_uhp_by_serpentinite(result, OUTPUT_DIR)
    build_figure_all_12_panels(result, OUTPUT_DIR)

    print("\nFinished successfully.")
    print(f"Monte Carlo iterations per metamafic stage and serpentinite scenario: {N_ITER_PER_ROW:,}")
    print("Serpentinite endmembers: forearc serpentinite + subarc serpentinite")
    print("Figure 1: 2x2 HP/UHP 95% regions; columns = forearc vs subarc serpentinite")
    print("Figure 2: 12 panels; two original-style 2x3 blocks, one per serpentinite endmember")
    print(f"All outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
