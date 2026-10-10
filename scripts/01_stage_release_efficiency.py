import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path

# ================== 0. CONFIGURATION ==================
INPUT_CSV = 'data/release_efficiency.csv'
N_ITER = 10000
RANDOM_SEED = 42

# Store the output in a dedicated results directory.
OUTPUT_DIR = Path('results/release_efficiency_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ROCK_TYPES = ['AOC', 'Blueschist', 'HP eclogite', 'UHP eclogite']
elements = ['F_ppm', 'Cl_ppm', 'Br_ppm', 'I_ppm']
element_labels = ['F', 'Cl', 'Br', 'I']

# ================== 1. LOAD AND PREPROCESS DATA ==================
df = pd.read_csv(INPUT_CSV)

print("Input data preview:")
print(df.head())
print("Column names:", df.columns.tolist())
print("Rock types in the dataset:", df['Rock_Type'].dropna().unique())

# -------- Unit conversion: ppb -> ppm --------
# Prefer existing Br_ppm / I_ppm columns when available;
# otherwise convert concentrations from ppb to ppm.
if 'Br_ppm' not in df.columns and 'Br_ppb' in df.columns:
    df['Br_ppm'] = pd.to_numeric(df['Br_ppb'], errors='coerce') / 1000.0
elif 'Br_ppm' in df.columns:
    df['Br_ppm'] = pd.to_numeric(df['Br_ppm'], errors='coerce')

if 'I_ppm' not in df.columns and 'I_ppb' in df.columns:
    df['I_ppm'] = pd.to_numeric(df['I_ppb'], errors='coerce') / 1000.0
elif 'I_ppm' in df.columns:
    df['I_ppm'] = pd.to_numeric(df['I_ppm'], errors='coerce')

# Check required input columns.
required_cols = ['Rock_Type'] + elements
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(
        f"Missing required input columns: {missing_cols}\n"
        f"Available columns: {df.columns.tolist()}"
    )

# Convert all elemental columns to numeric; blanks, NA and <DL become NaN.
for elem in elements:
    df[elem] = pd.to_numeric(df[elem], errors='coerce')

# Concentrations <= 0 are treated as missing for that element only;
# missing one element does not remove a sample from the other elements.
for elem in elements:
    df.loc[df[elem] <= 0, elem] = np.nan

print(f"\nTotal input samples: {len(df)}")
print("Note: samples are not discarded when one or more halogens are missing.")

# ================== 2. VALID DATA BY ROCK TYPE AND ELEMENT ==================
# Key principle:
# For example, if a blueschist sample lacks F but has Cl, Br and I:
#   - exclude it from the F bootstrap;
#   - retain it for the Cl, Br and I bootstraps.
rock_element_data = {
    rock: {
        elem: df.loc[df['Rock_Type'] == rock, elem].dropna().to_numpy(dtype=float)
        for elem in elements
    }
    for rock in ROCK_TYPES
}

# Diagnostics: valid sample counts and concentration ranges by rock type and element.
sample_summary_rows = []
print("\n================ VALID SAMPLES BY ELEMENT ================")
for rock in ROCK_TYPES:
    print(f"\n{rock}:")
    for elem, label in zip(elements, element_labels):
        values = rock_element_data[rock][elem]
        if len(values) > 0:
            print(
                f"  {label}: n={len(values)}, "
                f"min={np.min(values):.6g}, max={np.max(values):.6g}, "
                f"mean={np.mean(values):.6g}, median={np.median(values):.6g}"
            )
            sample_summary_rows.append({
                'Rock_Type': rock,
                'Element': label,
                'Column': elem,
                'n_valid': len(values),
                'min': np.min(values),
                'max': np.max(values),
                'mean': np.mean(values),
                'median': np.median(values)
            })
        else:
            print(f"  {label}: n=0  [WARNING: no valid values for this element in this stage]")
            sample_summary_rows.append({
                'Rock_Type': rock,
                'Element': label,
                'Column': elem,
                'n_valid': 0,
                'min': np.nan,
                'max': np.nan,
                'mean': np.nan,
                'median': np.nan
            })

sample_summary = pd.DataFrame(sample_summary_rows)
sample_summary.to_csv(
    OUTPUT_DIR / 'valid_sample_counts_by_rock_and_element.csv',
    index=False,
    encoding='utf-8-sig'
)

# ================== 3. ELEMENT-WISE MONTE CARLO BOOTSTRAP ==================
def bootstrap_mean(values, n_iter, rng):
    """
    Independently bootstrap valid observations of one element in one rock type,
    returning n_iter bootstrap sample means.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n = len(values)
    if n == 0:
        return np.full(n_iter, np.nan)

    # Preserve this element's valid sample count (n) in every bootstrap resample.
    idx = rng.integers(0, n, size=(n_iter, n))
    return values[idx].mean(axis=1)


def mc_stage_efficiencies(data_dict, n_iter=10000, random_seed=42):
    """
    Perform independent Monte Carlo bootstrapping for F, Cl, Br and I.

    Missing-data rules:
    - An observation missing an element is excluded for that element only;
    - Other measured elements from the same observation remain eligible;
    - Simultaneous availability of F, Cl, Br and I is not required.

    Stage contributions normalized to the AOC reference:
      stage1 = (AOC - BS) / AOC * 100
      stage2 = [(BS - HP) / BS] * (BS / AOC) * 100
             = (BS - HP) / AOC * 100
      stage3 = [(HP - UHP) / HP] * (HP / AOC) * 100
             = (HP - UHP) / AOC * 100

    Positive values indicate net apparent loss relative to AOC; negative values indicate apparent enrichment or metasomatism.
    """
    rng = np.random.default_rng(random_seed)
    dfs = {}

    for elem in elements:
        # Bootstrap each element independently for each rock type.
        mean_aoc = bootstrap_mean(data_dict['AOC'][elem], n_iter, rng)
        mean_bs = bootstrap_mean(data_dict['Blueschist'][elem], n_iter, rng)
        mean_hp = bootstrap_mean(data_dict['HP eclogite'][elem], n_iter, rng)
        mean_uhp = bootstrap_mean(data_dict['UHP eclogite'][elem], n_iter, rng)

        # If an element has no observations in a required stage, the resulting estimate is NaN.
        with np.errstate(divide='ignore', invalid='ignore'):
            eff1 = np.where(
                np.isfinite(mean_aoc) & (mean_aoc > 0),
                (mean_aoc - mean_bs) / mean_aoc,
                np.nan
            )

            eff2 = np.where(
                np.isfinite(mean_bs) & (mean_bs > 0),
                (mean_bs - mean_hp) / mean_bs,
                np.nan
            )

            eff3 = np.where(
                np.isfinite(mean_hp) & (mean_hp > 0),
                (mean_hp - mean_uhp) / mean_hp,
                np.nan
            )

        # Calculate stage contributions normalized to the initial AOC reference.
        remaining1 = 1.0 - eff1
        remaining2 = remaining1 * (1.0 - eff2)

        contrib1 = eff1 * 100.0
        contrib2 = eff2 * remaining1 * 100.0
        contrib3 = eff3 * remaining2 * 100.0

        df_elem = pd.DataFrame({
            'Blueschist (stage1)': contrib1,
            'HP Eclogite (stage2)': contrib2,
            'UHP Eclogite (stage3)': contrib3
        })

        dfs[elem] = df_elem

    return dfs


# ================== 4. RUN THE SIMULATION ==================
dfs_efficiencies = mc_stage_efficiencies(
    rock_element_data,
    n_iter=N_ITER,
    random_seed=RANDOM_SEED
)

# Print summary statistics.
for elem, label in zip(elements, element_labels):
    print(f"\n===== {label} normalized stage-release contribution (% of AOC) =====")
    df_elem = dfs_efficiencies[elem]

    if df_elem.notna().any().any():
        print(df_elem.describe(percentiles=[0.025, 0.16, 0.5, 0.84, 0.975]))
    else:
        print("No estimate is available because this element is missing from at least one required stage.")

# ================== 5. HELPERS: IQR FILTERING AND 95% INTERVALS ==================
def remove_outliers_iqr(data, factor=1.5):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    if len(data) < 4:
        return data

    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return data[(data >= lower) & (data <= upper)]


def robust_median_ci(series):
    arr = series.dropna().to_numpy(dtype=float)

    if len(arr) < 4:
        return np.nan, np.nan, np.nan

    arr_iqr = remove_outliers_iqr(arr)
    if len(arr_iqr) < 4:
        return np.nan, np.nan, np.nan

    median = np.median(arr_iqr)
    low_ci, high_ci = np.percentile(arr_iqr, [2.5, 97.5])
    return median, low_ci, high_ci


def box_iqr_count(series):
    # Count the observations within each boxplot's interquartile range (Q1 to Q3).
    arr = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan, 0

    q1, q3 = np.percentile(arr, [25, 75])
    n_in_box = int(np.sum((arr >= q1) & (arr <= q3)))
    return q1, q3, n_in_box


# ================== 6. VISUALIZATION ==================
ylim_dict = {
    'F': (-100, 100),
    'Cl': (-30, 100),
    'Br': (-30, 100),
    'I': (-30, 100)
}

bin_count_dict = {
    'F': 100,
    'Cl': 100,
    'Br': 100,
    'I': 200
}

fig, axes = plt.subplots(2, 2, figsize=(14, 14))
axes = axes.flatten()

colors = [
    (62/255, 158/255, 179/255),
    (241/255, 163/255, 97/255),
    (136/255, 176/255, 124/255)
]
stage_labels = [
    'AOC →\nBlueschist',
    'Blueschist →\nHP eclogite',
    'HP eclogite →\nUHP eclogite'
]

box_width = 0.25
hist_max_width = 0.35
box_offset = -0.2

for i, (elem, label) in enumerate(zip(elements, element_labels)):
    ax = axes[i]
    ax.set_box_aspect(1)

    df_elem = dfs_efficiencies[elem]
    data_raw = [
        df_elem[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        for col in df_elem.columns
    ]

    x_pos = np.arange(1, 4)
    bin_count = bin_count_dict[label]

    # Use consistent histogram bins within each panel.
    nonempty = [d for d in data_raw if len(d) > 0]
    if nonempty:
        all_data_sub = np.concatenate(nonempty)
        sub_min = np.min(all_data_sub)
        sub_max = np.max(all_data_sub)

        if np.isfinite(sub_min) and np.isfinite(sub_max) and sub_max > sub_min:
            sub_bins = np.linspace(sub_min, sub_max, bin_count + 1)
        else:
            center = sub_min if np.isfinite(sub_min) else 0.0
            sub_bins = np.array([center - 0.5, center + 0.5])
    else:
        sub_bins = np.array([0.0, 1.0])

    # Use [nan] as a placeholder for empty groups in Matplotlib boxplots.
    boxplot_data = [d if len(d) > 0 else np.array([np.nan]) for d in data_raw]

    bp = ax.boxplot(
        boxplot_data,
        positions=x_pos + box_offset,
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        zorder=5,
        boxprops=dict(facecolor='lightgray', edgecolor='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        medianprops=dict(color='red', linewidth=1.5)
    )

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Horizontal histograms and kernel density estimates (KDE).
    for j, (series, pos) in enumerate(zip(data_raw, x_pos)):
        if len(series) < 2:
            continue

        hist, bin_edges = np.histogram(series, bins=sub_bins)
        if hist.max() > 0:
            hist_norm = hist / hist.max() * hist_max_width
        else:
            hist_norm = np.zeros_like(hist, dtype=float)

        heights = np.diff(bin_edges)
        ax.barh(
            bin_edges[:-1],
            hist_norm,
            height=heights,
            left=pos,
            align='edge',
            color=colors[j],
            alpha=0.5,
            edgecolor='none'
        )

        # Compute the KDE only when observations vary, avoiding singular covariance errors.
        if len(np.unique(series)) > 1 and np.std(series) > 0:
            try:
                kde = gaussian_kde(series)
                y_vals = np.linspace(series.min(), series.max(), 200)
                density = kde(y_vals)

                if np.max(density) > 0:
                    density_norm = density / np.max(density) * hist_max_width
                    ax.plot(
                        pos + density_norm,
                        y_vals,
                        color=colors[j],
                        linewidth=1.5,
                        zorder=10
                    )
            except np.linalg.LinAlgError:
                pass

    # Robust median labels and observation counts inside the boxes.
    medians = []
    box_stats = []
    for series in data_raw:
        if len(series) >= 4:
            med, _, _ = robust_median_ci(pd.Series(series))
            medians.append(med)
        else:
            medians.append(np.nan)

        q1, q3, n_in_box = box_iqr_count(series)
        box_stats.append((q1, q3, n_in_box))

    y_low, y_high = ylim_dict[label]
    ax.set_ylim(y_low, y_high)
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    y_range = y_high - y_low
    text_offset = y_range * 0.08

    # Annotate median values.
    for xi, med, col in zip(x_pos, medians, colors):
        if np.isfinite(med):
            ax.text(
                xi + box_offset,
                med + text_offset,
                f'{med:.2f}%',
                ha='center',
                va='bottom',
                fontsize=9,
                color=col,
                bbox=dict(
                    boxstyle='round,pad=0.2',
                    facecolor='white',
                    alpha=0.8,
                    edgecolor='none'
                )
            )

    # Annotate the number of observations within each box (Q1–Q3).
    for xi, (q1, q3, n_in_box), col in zip(x_pos, box_stats, colors):
        if np.isfinite(q1) and np.isfinite(q3) and n_in_box > 0:
            y_center = (q1 + q3) / 2.0
            ax.text(
                xi + box_offset,
                y_center,
                f'n={n_in_box}',
                ha='center',
                va='center',
                fontsize=8,
                color='black',
                bbox=dict(
                    boxstyle='round,pad=0.2',
                    facecolor='white',
                    alpha=0.85,
                    edgecolor=col,
                    linewidth=0.8
                ),
                zorder=12
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(stage_labels)
    ax.set_title(f'{label} release contribution (% of AOC)', fontsize=14)
    ax.set_ylabel(
        r'$\frac{C_{\mathrm{prev}}-C_{\mathrm{next}}}'
        r'{C_{\mathrm{AOC}}}\times100\%$'
    )

plt.tight_layout(pad=2.0)

# Save the figures and display them.
fig.savefig(OUTPUT_DIR / 'halogen_stage_release_contributions.svg', bbox_inches='tight')
fig.savefig(OUTPUT_DIR / 'halogen_stage_release_contributions.pdf', bbox_inches='tight')
plt.show()

# ================== 7. EXPORT RESULTS ==================
for elem, label in zip(elements, element_labels):
    dfs_efficiencies[elem].to_csv(
        OUTPUT_DIR / f'{label}_stage_contributions.csv',
        index=False,
        encoding='utf-8-sig'
    )

# Compile the summary statistics.
summary_rows = []
for elem, label in zip(elements, element_labels):
    df_elem = dfs_efficiencies[elem]

    for col in df_elem.columns:
        arr = df_elem[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)

        if len(arr) == 0:
            stats = {
                'N_MC_valid': 0,
                'mean': np.nan,
                'median': np.nan,
                'p2.5': np.nan,
                'p16': np.nan,
                'p84': np.nan,
                'p97.5': np.nan
            }
        else:
            p2_5, p16, p50, p84, p97_5 = np.percentile(arr, [2.5, 16, 50, 84, 97.5])
            stats = {
                'N_MC_valid': len(arr),
                'mean': np.mean(arr),
                'median': p50,
                'p2.5': p2_5,
                'p16': p16,
                'p84': p84,
                'p97.5': p97_5
            }

        summary_rows.append({
            'Element': label,
            'Stage': col,
            **stats
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    OUTPUT_DIR / 'stage_release_summary.csv',
    index=False,
    encoding='utf-8-sig'
)

print("\n================ CALCULATION COMPLETE ================")
print(f"Output directory: {OUTPUT_DIR}")
print("\nKey rules used in this version:")
print("1. Samples are not required to have all four halogen measurements.")
print("2. Each halogen is calculated independently using its available observations.")
print("3. Missing one element does not exclude measured values of other elements.")
print("4. Valid sample counts can differ by element and metamorphic stage.")
