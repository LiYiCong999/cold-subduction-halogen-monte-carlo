# -*- coding: utf-8 -*-
"""
NC-oriented path-constrained Monte Carlo model for progressive open-system dehydration
AOC -> BS -> HP -> UHP

Purpose
-------
Use measured whole-rock halogen compositions and sample-specific H2O ranges to infer
stage-resolved halogen loss and the conditional composition of released fluids.

Core principles
---------------
1. Each realization jointly samples one AOC, BS, HP and UHP natural sample.
2. H2O is sampled from each sample's own [min, max] range.
3. A path is physically admissible only if:
       H2O_AOC >= H2O_BS >= H2O_HP >= H2O_UHP
4. Initial AOC total mass = 1 kg. Anhydrous rock mass is conserved along the whole path.
5. No external fluid/solid input is imposed in the forward mass balance.
6. Fluids released at each stage are immediately removed (open-system dehydration).
7. Missing F/Cl/Br/I values remain missing; no interpolation, no mean filling, no zero filling.
8. Element behavior is NOT called accepted/rejected. For H2O-valid paths it is classified as:
       net_loss
       near_zero_retention
       apparent_gain
       missing
       no_fluid_release
9. Fluid compositions are reported ONLY conditional on net elemental loss:
       C_fluid | Δm_X > 0
10. Halogen ratios are reported ONLY for joint-net-loss realizations of both elements.
11. Main outputs explicitly report:
       - physically admissible H2O-path fraction
       - P(net loss | H2O-valid and element evaluable)
       - conditional fluid-composition distributions
       - P(joint net loss | pair evaluable) for ratios
12. No mantle mixing is performed: figures show released-fluid halogen ratios only.
13. Optional diagnostics include:
       - Monte Carlo convergence using prefixes of the main simulation
       - H2O sampling sensitivity: uniform vs triangular(midpoint mode)

Input CSV columns
-----------------
stage,sample_id,H2O_min_wt%,H2O_max_wt%,F_ppm,Cl_ppm,Br_ppb,I_ppb

Recommended stage names
-----------------------
AOC
BS
HP
UHP

Dependencies
------------
pip install numpy pandas matplotlib scipy openpyxl
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 1. CONFIG
# =============================================================================

# Modify this path for your computer.
INPUT_CSV = "data/bulk_data.csv"

# Relative path: output folder will be created under the current working directory.
OUTPUT_DIR = "results/MC_halogen_dehydration_NC"

# Main Monte Carlo size.
N_SIM = 100_000
RANDOM_SEED = 20260917
CHUNK_SIZE = 50_000

# Main H2O sampling distribution:
# "uniform"    : U(min, max)
# "triangular" : triangular(min, midpoint, max)
H2O_SAMPLING_MAIN = "uniform"

# Optional sensitivity test.
RUN_H2O_SENSITIVITY = True
H2O_SENSITIVITY_MODES = ("uniform", "triangular")
N_SENSITIVITY = 100_000
SENSITIVITY_SEED = 20260918

# Optional convergence analysis based on prefixes of the main simulation.
RUN_CONVERGENCE = True
CONVERGENCE_ATTEMPTS = (10_000, 50_000, 100_000)

# Relative numerical tolerance used ONLY to distinguish near-zero mass change
# from positive/negative mass change. It is not a geological cutoff.
ELEMENT_CHANGE_REL_TOL = 1e-10

SAVE_ALL_H2O_VALID_PATHS = True
SAVE_H2O_INADMISSIBLE_PATHS = True

# Diagnostics for very small stage fluid fractions.
# These realizations are not silently deleted.
FLUID_FRACTION_DIAGNOSTIC_THRESHOLDS = (1e-4, 1e-3, 5e-3, 1e-2)

PLOT_DPI = 600

# Six-panel fluid-composition figure: NO mantle endmember or reservoir mixing.
# Each point consists only of modeled halogens released during one stage.
MAKE_FLUID_6PANEL_FIGURE = True
FLUID_PLOT_XLIM = (1e-6, 1e-1)  # I/Cl (mass ratio)
FLUID_PLOT_FYLIM = (1e-4, 1e2)  # F/Cl (mass ratio)
FLUID_PLOT_BRYLIM = (1e-4, 1e-1)  # Br/Cl (mass ratio)
# Use fixed axes requested by the user for direct visual comparability.
AUTO_EXPAND_FLUID_AXES = False
FLUID_KDE_MAX_POINTS = 1600  # plotting only; statistics use all valid paths
FLUID_KDE_GRID = 90
FLUID_HIST_BINS = 20
# Colors taken from the previously supplied stage-based visualization code.
STAGE_COLORS = {"AOC_BS": (62/255, 158/255, 179/255),
                "BS_HP": (241/255, 163/255, 97/255),
                "HP_UHP": (136/255, 176/255, 124/255)}

# Publication-style defaults.
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

STAGE_LABELS = {
    "AOC_BS": "AOC→BS",
    "BS_HP": "BS→HP",
    "HP_UHP": "HP→UHP",
}

STAGE_PAIRS = {
    "AOC_BS": ("AOC", "BS"),
    "BS_HP": ("BS", "HP"),
    "HP_UHP": ("HP", "UHP"),
}

ELEMENTS = {
    "F": "ppm",
    "Cl": "ppm",
    "Br": "ppb",
    "I": "ppb",
}

ELEMENT_COLS = {
    "F": "F_ppm",
    "Cl": "Cl_ppm",
    "Br": "Br_ppb",
    "I": "I_ppb",
}

ATOMIC_WEIGHTS = {
    "F": 18.998403163,
    "Cl": 35.45,
    "Br": 79.904,
    "I": 126.90447,
}

RATIOS = [
    ("F", "Cl"),
    ("Br", "Cl"),
    ("I", "Cl"),
]

REQUIRED_COLUMNS = [
    "stage",
    "sample_id",
    "H2O_min_wt%",
    "H2O_max_wt%",
    "F_ppm",
    "Cl_ppm",
    "Br_ppb",
    "I_ppb",
]

VALID_STAGES = ("AOC", "BS", "HP", "UHP")


# =============================================================================
# 2. INPUT AND VALIDATION
# =============================================================================

def clean_stage_name(x: str) -> str:
    """Normalize common stage spellings to AOC/BS/HP/UHP."""
    if pd.isna(x):
        return np.nan

    s = str(x).strip().upper().replace("_", " ")
    s = " ".join(s.split())

    mapping = {
        "AOC": "AOC",
        "ALTERED OCEANIC CRUST": "AOC",

        "BS": "BS",
        "BLUESCHIST": "BS",
        "BLUESCHISTS": "BS",
        "BLUSCHIST": "BS",     # tolerate the common typo in existing table
        "BLUSCHISTS": "BS",

        "HP": "HP",
        "HP ECLOGITE": "HP",
        "HP ECLOGITES": "HP",
        "HIGH PRESSURE ECLOGITE": "HP",
        "HIGH PRESSURE ECLOGITES": "HP",
        "HIGH-PRESSURE ECLOGITE": "HP",
        "HIGH-PRESSURE ECLOGITES": "HP",

        "UHP": "UHP",
        "UHP ECLOGITE": "UHP",
        "UHP ECLOGITES": "UHP",
        "ULTRAHIGH PRESSURE ECLOGITE": "UHP",
        "ULTRAHIGH PRESSURE ECLOGITES": "UHP",
        "ULTRAHIGH-PRESSURE ECLOGITE": "UHP",
        "ULTRAHIGH-PRESSURE ECLOGITES": "UHP",
        "ULTRA-HIGH-PRESSURE ECLOGITE": "UHP",
        "ULTRA-HIGH-PRESSURE ECLOGITES": "UHP",
    }
    return mapping.get(s, s)


def load_and_validate_input(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            "Input CSV is missing required columns:\n"
            + "\n".join(missing_cols)
            + "\n\nRequired header:\n"
            + ",".join(REQUIRED_COLUMNS)
        )

    df = df[REQUIRED_COLUMNS].copy()
    df["stage"] = df["stage"].map(clean_stage_name)

    unknown_stages = sorted(set(df["stage"].dropna()) - set(VALID_STAGES))
    if unknown_stages:
        raise ValueError(
            f"Unrecognized stage values: {unknown_stages}\n"
            f"Please use {VALID_STAGES}."
        )

    for c in REQUIRED_COLUMNS[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    bad_h2o = (
        df["H2O_min_wt%"].isna()
        | df["H2O_max_wt%"].isna()
        | (df["H2O_min_wt%"] < 0)
        | (df["H2O_max_wt%"] < 0)
        | (df["H2O_min_wt%"] > df["H2O_max_wt%"])
        | (df["H2O_max_wt%"] >= 100)
    )
    if bad_h2o.any():
        bad = df.loc[
            bad_h2o,
            ["stage", "sample_id", "H2O_min_wt%", "H2O_max_wt%"],
        ]
        raise ValueError(
            "Invalid H2O ranges. Require 0 <= min <= max < 100 wt%:\n"
            + bad.to_string(index=False)
        )

    counts = df.groupby("stage").size()
    missing_stage = [s for s in VALID_STAGES if counts.get(s, 0) == 0]
    if missing_stage:
        raise ValueError(f"No samples found for stages: {missing_stage}")

    for c in ELEMENT_COLS.values():
        neg = df[c] < 0
        if neg.any():
            warnings.warn(f"{c} contains negative concentrations; converting them to NaN.")
            df.loc[neg, c] = np.nan

    return df.reset_index(drop=True)


def build_input_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage in VALID_STAGES:
        sub = df[df["stage"] == stage]
        row = {
            "stage": stage,
            "N_samples": len(sub),
            "H2O_min_global_wt%": sub["H2O_min_wt%"].min(),
            "H2O_max_global_wt%": sub["H2O_max_wt%"].max(),
        }
        for elem, col in ELEMENT_COLS.items():
            row[f"{elem}_measured_N"] = int(sub[col].notna().sum())
            row[f"{elem}_missing_N"] = int(sub[col].isna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 3. RANDOM SAMPLING AND MASS-BALANCE CORE
# =============================================================================

def draw_rows(
    stage_df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Equal-probability bootstrap sampling of natural samples within a stage."""
    idx = rng.integers(0, len(stage_df), size=n)
    return stage_df.iloc[idx].reset_index(drop=True)


def draw_h2o(
    sample_df: pd.DataFrame,
    rng: np.random.Generator,
    mode: str,
) -> np.ndarray:
    """
    Sample H2O from each sample-specific range and convert wt% to mass fraction.
    """
    low = sample_df["H2O_min_wt%"].to_numpy(dtype=float)
    high = sample_df["H2O_max_wt%"].to_numpy(dtype=float)

    if mode == "uniform":
        draw = rng.uniform(low, high)

    elif mode == "triangular":
        draw = np.empty(len(low), dtype=float)
        equal = np.isclose(low, high, rtol=0, atol=0)
        draw[equal] = low[equal]
        if (~equal).any():
            mode_mid = (low[~equal] + high[~equal]) / 2.0
            draw[~equal] = rng.triangular(
                low[~equal], mode_mid, high[~equal]
            )
    else:
        raise ValueError(
            f"Unknown H2O sampling mode {mode!r}. "
            "Use 'uniform' or 'triangular'."
        )

    return draw / 100.0


def rock_mass_from_dry_mass(
    dry_mass: np.ndarray,
    water_mass_fraction: np.ndarray,
) -> np.ndarray:
    """
    M_rock = M_dry / (1 - w)

    Initial AOC total rock mass is normalized to 1 kg.
    """
    return dry_mass / (1.0 - water_mass_fraction)


def concentration_to_mg_per_kg(values: np.ndarray, unit: str) -> np.ndarray:
    # ppm = mg/kg; ppb = ug/kg = 1e-3 mg/kg
    if unit == "ppm":
        return values
    if unit == "ppb":
        return values / 1000.0
    raise ValueError(unit)


def mg_per_kg_to_original_unit(values: np.ndarray, unit: str) -> np.ndarray:
    if unit == "ppm":
        return values
    if unit == "ppb":
        return values * 1000.0
    raise ValueError(unit)


def calc_stage_element(
    c_up_raw: np.ndarray,
    c_down_raw: np.ndarray,
    unit: str,
    m_up: np.ndarray,
    m_down: np.ndarray,
    m_fluid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute stage-specific elemental inventory change.

    Δm_X = C_up * M_up - C_down * M_down

    Status:
        missing
        no_fluid_release
        net_loss
        near_zero_retention
        apparent_gain

    The near-zero threshold is only a numerical tolerance:
        tol = ELEMENT_CHANGE_REL_TOL * max(|m_up_X|, |m_down_X|, 1e-15)

    Fluid concentration is calculated ONLY when status == net_loss.
    """
    c_up = concentration_to_mg_per_kg(c_up_raw, unit)
    c_down = concentration_to_mg_per_kg(c_down_raw, unit)

    n = len(m_fluid)
    loss = np.full(n, np.nan, dtype=float)
    fluid_conc = np.full(n, np.nan, dtype=float)
    status = np.full(n, "missing", dtype=object)

    concentration_available = np.isfinite(c_up) & np.isfinite(c_down)
    positive_fluid = m_fluid > 0

    no_fluid = concentration_available & (~positive_fluid)
    status[no_fluid] = "no_fluid_release"

    evaluable = concentration_available & positive_fluid
    if evaluable.any():
        inv_up = c_up[evaluable] * m_up[evaluable]
        inv_down = c_down[evaluable] * m_down[evaluable]
        delta = inv_up - inv_down
        loss[evaluable] = delta

        scale = np.maximum.reduce([
            np.abs(inv_up),
            np.abs(inv_down),
            np.full(len(inv_up), 1e-15),
        ])
        tol = ELEMENT_CHANGE_REL_TOL * scale

        net = delta > tol
        gain = delta < -tol
        near = ~(net | gain)

        idx = np.where(evaluable)[0]

        status[idx[net]] = "net_loss"
        status[idx[near]] = "near_zero_retention"
        status[idx[gain]] = "apparent_gain"

        if net.any():
            fluid_mgkg = delta[net] / m_fluid[idx[net]]
            fluid_conc[idx[net]] = mg_per_kg_to_original_unit(
                fluid_mgkg, unit
            )

    return loss, fluid_conc, status


def ratio_from_element_status(
    num_loss_mg: np.ndarray,
    den_loss_mg: np.ndarray,
    num_status: np.ndarray,
    den_status: np.ndarray,
    numerator: str,
    denominator: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate ratios only for joint-net-loss realizations.

    Ratio status:
        missing
        no_fluid_release
        joint_net_loss
        non_joint_net_loss
    """
    n = len(num_loss_mg)
    mass_ratio = np.full(n, np.nan, dtype=float)
    molar_ratio = np.full(n, np.nan, dtype=float)
    status = np.full(n, "missing", dtype=object)

    num_missing = num_status == "missing"
    den_missing = den_status == "missing"
    pair_missing = num_missing | den_missing

    pair_no_fluid = (
        (~pair_missing)
        & (
            (num_status == "no_fluid_release")
            | (den_status == "no_fluid_release")
        )
    )
    status[pair_no_fluid] = "no_fluid_release"

    joint = (
        (num_status == "net_loss")
        & (den_status == "net_loss")
    )
    status[joint] = "joint_net_loss"

    pair_evaluable = (~pair_missing) & (~pair_no_fluid)
    non_joint = pair_evaluable & (~joint)
    status[non_joint] = "non_joint_net_loss"

    if joint.any():
        mass_ratio[joint] = num_loss_mg[joint] / den_loss_mg[joint]
        molar_ratio[joint] = (
            num_loss_mg[joint] / ATOMIC_WEIGHTS[numerator]
        ) / (
            den_loss_mg[joint] / ATOMIC_WEIGHTS[denominator]
        )

    return mass_ratio, molar_ratio, status


def simulate_chunk(
    stage_dfs: Dict[str, pd.DataFrame],
    n: int,
    rng: np.random.Generator,
    global_start_id: int,
    h2o_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate one chunk.

    Returns
    -------
    admissible_df
        H2O-monotonic whole paths with stage-by-stage mass balance.
    inadmissible_df
        Paths rejected ONLY because H2O_AOC >= BS >= HP >= UHP was violated.
    """
    draws = {
        s: draw_rows(stage_dfs[s], n, rng)
        for s in VALID_STAGES
    }
    h2o = {
        s: draw_h2o(draws[s], rng, h2o_mode)
        for s in VALID_STAGES
    }

    path_id = np.arange(
        global_start_id,
        global_start_id + n,
        dtype=np.int64,
    )

    physically_admissible = (
        (h2o["AOC"] >= h2o["BS"])
        & (h2o["BS"] >= h2o["HP"])
        & (h2o["HP"] >= h2o["UHP"])
    )

    inad = pd.DataFrame({
        "path_id": path_id[~physically_admissible],
        "AOC_sample_id": draws["AOC"].loc[
            ~physically_admissible, "sample_id"
        ].astype(str).to_numpy(),
        "BS_sample_id": draws["BS"].loc[
            ~physically_admissible, "sample_id"
        ].astype(str).to_numpy(),
        "HP_sample_id": draws["HP"].loc[
            ~physically_admissible, "sample_id"
        ].astype(str).to_numpy(),
        "UHP_sample_id": draws["UHP"].loc[
            ~physically_admissible, "sample_id"
        ].astype(str).to_numpy(),
        "H2O_AOC_wt%": h2o["AOC"][~physically_admissible] * 100,
        "H2O_BS_wt%": h2o["BS"][~physically_admissible] * 100,
        "H2O_HP_wt%": h2o["HP"][~physically_admissible] * 100,
        "H2O_UHP_wt%": h2o["UHP"][~physically_admissible] * 100,
        "path_status": "physically_inadmissible_H2O_path",
        "H2O_sampling_mode": h2o_mode,
    })

    if not physically_admissible.any():
        return pd.DataFrame(), inad

    pid = path_id[physically_admissible]
    h = {
        s: h2o[s][physically_admissible]
        for s in VALID_STAGES
    }
    d = {
        s: draws[s].loc[physically_admissible].reset_index(drop=True)
        for s in VALID_STAGES
    }

    # Unified mass basis: initial AOC = 1 kg
    m_aoc = np.ones(len(pid), dtype=float)
    dry_mass = 1.0 - h["AOC"]

    m_bs = rock_mass_from_dry_mass(dry_mass, h["BS"])
    m_hp = rock_mass_from_dry_mass(dry_mass, h["HP"])
    m_uhp = rock_mass_from_dry_mass(dry_mass, h["UHP"])

    rock_masses = {
        "AOC": m_aoc,
        "BS": m_bs,
        "HP": m_hp,
        "UHP": m_uhp,
    }

    fluid_masses = {}
    for skey, (up, down) in STAGE_PAIRS.items():
        fluid_masses[skey] = rock_masses[up] - rock_masses[down]

    out = pd.DataFrame({
        "path_id": pid,
        "path_status": "physically_admissible_H2O_path",
        "H2O_sampling_mode": h2o_mode,

        "AOC_sample_id": d["AOC"]["sample_id"].astype(str),
        "BS_sample_id": d["BS"]["sample_id"].astype(str),
        "HP_sample_id": d["HP"]["sample_id"].astype(str),
        "UHP_sample_id": d["UHP"]["sample_id"].astype(str),

        "H2O_AOC_wt%": h["AOC"] * 100,
        "H2O_BS_wt%": h["BS"] * 100,
        "H2O_HP_wt%": h["HP"] * 100,
        "H2O_UHP_wt%": h["UHP"] * 100,

        "M_AOC_kg_per_1kgAOC": m_aoc,
        "M_BS_kg_per_1kgAOC": m_bs,
        "M_HP_kg_per_1kgAOC": m_hp,
        "M_UHP_kg_per_1kgAOC": m_uhp,
    })

    # Accumulate all derived columns first, then attach them as ONE DataFrame.
    # Repeated column-by-column insertion fragments pandas' internal storage.
    derived_columns = {}

    for skey, (up, down) in STAGE_PAIRS.items():
        mf = fluid_masses[skey]
        derived_columns[f"{skey}_fluid_mass_kg_per_1kgAOC"] = mf
        derived_columns[f"{skey}_fluid_fraction_of_initial_AOC"] = mf
        derived_columns[f"{skey}_fluid_fraction_of_stage_start"] = (
            mf / rock_masses[up]
        )

    losses: Dict[Tuple[str, str], np.ndarray] = {}
    statuses: Dict[Tuple[str, str], np.ndarray] = {}

    for skey, (up, down) in STAGE_PAIRS.items():
        for elem, unit in ELEMENTS.items():
            c_up = d[up][ELEMENT_COLS[elem]].to_numpy(dtype=float)
            c_down = d[down][ELEMENT_COLS[elem]].to_numpy(dtype=float)

            loss, fluid_c, status = calc_stage_element(
                c_up_raw=c_up,
                c_down_raw=c_down,
                unit=unit,
                m_up=rock_masses[up],
                m_down=rock_masses[down],
                m_fluid=fluid_masses[skey],
            )

            losses[(skey, elem)] = loss
            statuses[(skey, elem)] = status

            # First-code-compatible concentration SIGN diagnostic on the SAME
            # sampled pair. Do not confuse this with the mass-inventory change.
            # Concentration units follow ELEMENTS (ppm for F/Cl, ppb for Br/I).
            derived_columns[f"{skey}_{elem}_concentration_difference_{unit}"] = c_up - c_down
            derived_columns[f"{skey}_{elem}_loss_mg_per_1kgAOC"] = loss
            derived_columns[f"{skey}_{elem}_fluid_{unit}_conditional_net_loss"] = fluid_c
            derived_columns[f"{skey}_{elem}_process_status"] = status

        for num, den in RATIOS:
            mass_r, molar_r, ratio_status = ratio_from_element_status(
                num_loss_mg=losses[(skey, num)],
                den_loss_mg=losses[(skey, den)],
                num_status=statuses[(skey, num)],
                den_status=statuses[(skey, den)],
                numerator=num,
                denominator=den,
            )
            derived_columns[f"{skey}_{num}_{den}_mass_ratio_conditional_joint_net_loss"] = mass_r
            derived_columns[f"{skey}_{num}_{den}_molar_ratio_conditional_joint_net_loss"] = molar_r
            derived_columns[f"{skey}_{num}_{den}_ratio_status"] = ratio_status

    # Whole-path mass-balance diagnostics
    derived_columns["total_fluid_mass_kg_per_1kgAOC"] = (
        fluid_masses["AOC_BS"]
        + fluid_masses["BS_HP"]
        + fluid_masses["HP_UHP"]
    )
    derived_columns["fluid_mass_balance_residual_kg"] = (
        derived_columns["total_fluid_mass_kg_per_1kgAOC"]
        - (m_aoc - m_uhp)
    )

    # Element telescoping diagnostic, only when every stage loss is finite.
    for elem, unit in ELEMENTS.items():
        c_aoc = concentration_to_mg_per_kg(
            d["AOC"][ELEMENT_COLS[elem]].to_numpy(dtype=float),
            unit,
        )
        c_uhp = concentration_to_mg_per_kg(
            d["UHP"][ELEMENT_COLS[elem]].to_numpy(dtype=float),
            unit,
        )
        direct_total = c_aoc * m_aoc - c_uhp * m_uhp
        summed_stages = (
            losses[("AOC_BS", elem)]
            + losses[("BS_HP", elem)]
            + losses[("HP_UHP", elem)]
        )
        residual = np.full(len(pid), np.nan, dtype=float)
        ok = np.isfinite(direct_total) & np.isfinite(summed_stages)
        residual[ok] = summed_stages[ok] - direct_total[ok]
        derived_columns[f"{elem}_mass_balance_residual_mg"] = residual

    # The base columns and all derived columns are joined in one batch.
    # Numerical values and geological acceptance criteria are unchanged.
    out = pd.concat(
        [out, pd.DataFrame(derived_columns, index=out.index)],
        axis=1,
        copy=False,
    )
    return out, inad


def run_simulation(
    df: pd.DataFrame,
    n_sim: int,
    seed: int,
    h2o_mode: str,
    chunk_size: int = CHUNK_SIZE,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stage_dfs = {
        s: df[df["stage"] == s].reset_index(drop=True)
        for s in VALID_STAGES
    }
    rng = np.random.default_rng(seed)

    admissible_chunks: List[pd.DataFrame] = []
    inadmissible_chunks: List[pd.DataFrame] = []

    attempted = 0
    while attempted < n_sim:
        n = min(chunk_size, n_sim - attempted)
        good, bad = simulate_chunk(
            stage_dfs=stage_dfs,
            n=n,
            rng=rng,
            global_start_id=attempted + 1,
            h2o_mode=h2o_mode,
        )
        if len(good):
            admissible_chunks.append(good)
        if len(bad):
            inadmissible_chunks.append(bad)

        attempted += n
        if verbose:
            print(
                f"[{h2o_mode}] completed "
                f"{attempted:,}/{n_sim:,} attempted paths"
            )

    admissible = (
        pd.concat(admissible_chunks, ignore_index=True)
        if admissible_chunks else pd.DataFrame()
    )
    inadmissible = (
        pd.concat(inadmissible_chunks, ignore_index=True)
        if inadmissible_chunks else pd.DataFrame()
    )

    return admissible, inadmissible


# =============================================================================
# 4. SUMMARY FUNCTIONS
# =============================================================================

def finite_array(series: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def numeric_summary(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return {
            "N": 0,
            "median": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "P2.5": np.nan,
            "P16": np.nan,
            "P84": np.nan,
            "P97.5": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "N": len(arr),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan,
        "P2.5": float(np.percentile(arr, 2.5)),
        "P16": float(np.percentile(arr, 16)),
        "P84": float(np.percentile(arr, 84)),
        "P97.5": float(np.percentile(arr, 97.5)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def build_h2o_path_summary(
    admissible_df: pd.DataFrame,
    inadmissible_df: pd.DataFrame,
    n_attempted: int,
    mode: str,
) -> pd.DataFrame:
    n_good = len(admissible_df)
    n_bad = len(inadmissible_df)

    return pd.DataFrame([{
        "H2O_sampling_mode": mode,
        "attempted_paths": n_attempted,
        "physically_admissible_paths": n_good,
        "physically_inadmissible_paths": n_bad,
        "P_physically_admissible": (
            n_good / n_attempted if n_attempted else np.nan
        ),
        "P_physically_inadmissible": (
            n_bad / n_attempted if n_attempted else np.nan
        ),
        "constraint": "H2O_AOC >= H2O_BS >= H2O_HP >= H2O_UHP",
    }])


def build_element_process_summary(
    admissible_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Main support metric:
        P_net_loss_given_evaluable
        = N_net_loss /
          (N_net_loss + N_near_zero_retention + N_apparent_gain)

    Missing data and no-fluid stage realizations are shown separately and are
    not silently included in the denominator.
    """
    rows = []

    for skey, slabel in STAGE_LABELS.items():
        for elem in ELEMENTS:
            s = admissible_df[f"{skey}_{elem}_process_status"]

            counts = s.value_counts(dropna=False)
            n_total = len(s)
            n_missing = int(counts.get("missing", 0))
            n_no_fluid = int(counts.get("no_fluid_release", 0))
            n_net = int(counts.get("net_loss", 0))
            n_ret = int(counts.get("near_zero_retention", 0))
            n_gain = int(counts.get("apparent_gain", 0))

            n_evaluable = n_net + n_ret + n_gain

            rows.append({
                "stage": slabel,
                "stage_key": skey,
                "element": elem,
                "N_H2O_valid_paths": n_total,
                "N_missing": n_missing,
                "N_no_fluid_release": n_no_fluid,
                "N_element_evaluable": n_evaluable,
                "N_net_loss": n_net,
                "N_near_zero_retention": n_ret,
                "N_apparent_gain": n_gain,
                "P_element_evaluable_given_H2O_valid": (
                    n_evaluable / n_total if n_total else np.nan
                ),
                "P_net_loss_given_evaluable": (
                    n_net / n_evaluable if n_evaluable else np.nan
                ),
                "P_near_zero_retention_given_evaluable": (
                    n_ret / n_evaluable if n_evaluable else np.nan
                ),
                "P_apparent_gain_given_evaluable": (
                    n_gain / n_evaluable if n_evaluable else np.nan
                ),
                "interpretation_note": (
                    "Probabilities are conditional on physically admissible "
                    "H2O paths and available/evaluable element data."
                ),
            })

    return pd.DataFrame(rows)


def build_conditional_fluid_summary(
    admissible_df: pd.DataFrame,
    process_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    probability_lookup = {
        (r["stage_key"], r["element"]): r["P_net_loss_given_evaluable"]
        for _, r in process_df.iterrows()
    }

    for skey, slabel in STAGE_LABELS.items():
        for elem, unit in ELEMENTS.items():
            col = f"{skey}_{elem}_fluid_{unit}_conditional_net_loss"
            arr = finite_array(admissible_df[col])
            st = numeric_summary(arr)

            rows.append({
                "stage": slabel,
                "stage_key": skey,
                "element": elem,
                "unit": unit,
                "conditioning": "net_loss only (Δm_X > numerical tolerance)",
                "P_net_loss_given_evaluable": probability_lookup[(skey, elem)],
                **st,
            })

    return pd.DataFrame(rows)


def build_ratio_process_summary(
    admissible_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for skey, slabel in STAGE_LABELS.items():
        for num, den in RATIOS:
            s = admissible_df[f"{skey}_{num}_{den}_ratio_status"]
            counts = s.value_counts(dropna=False)

            n_total = len(s)
            n_missing = int(counts.get("missing", 0))
            n_no_fluid = int(counts.get("no_fluid_release", 0))
            n_joint = int(counts.get("joint_net_loss", 0))
            n_non_joint = int(counts.get("non_joint_net_loss", 0))
            n_pair_evaluable = n_joint + n_non_joint

            rows.append({
                "stage": slabel,
                "stage_key": skey,
                "ratio": f"{num}/{den}",
                "N_H2O_valid_paths": n_total,
                "N_pair_missing": n_missing,
                "N_no_fluid_release": n_no_fluid,
                "N_pair_evaluable": n_pair_evaluable,
                "N_joint_net_loss": n_joint,
                "N_non_joint_net_loss": n_non_joint,
                "P_joint_net_loss_given_pair_evaluable": (
                    n_joint / n_pair_evaluable
                    if n_pair_evaluable else np.nan
                ),
                "interpretation_note": (
                    "Ratio distributions are conditional on both numerator "
                    "and denominator showing net loss in the same realization."
                ),
            })

    return pd.DataFrame(rows)


def build_conditional_ratio_summary(
    admissible_df: pd.DataFrame,
    ratio_process_df: pd.DataFrame,
) -> pd.DataFrame:
    p_lookup = {
        (r["stage_key"], r["ratio"]):
            r["P_joint_net_loss_given_pair_evaluable"]
        for _, r in ratio_process_df.iterrows()
    }

    rows = []
    for skey, slabel in STAGE_LABELS.items():
        for num, den in RATIOS:
            ratio_name = f"{num}/{den}"
            for ratio_type in ("mass", "molar"):
                col = (
                    f"{skey}_{num}_{den}_{ratio_type}_ratio_"
                    "conditional_joint_net_loss"
                )
                arr = finite_array(admissible_df[col])
                st = numeric_summary(arr)

                rows.append({
                    "stage": slabel,
                    "stage_key": skey,
                    "ratio": ratio_name,
                    "ratio_type": ratio_type,
                    "unit": "dimensionless",
                    "conditioning": "joint_net_loss only",
                    "P_joint_net_loss_given_pair_evaluable":
                        p_lookup[(skey, ratio_name)],
                    **st,
                })

    return pd.DataFrame(rows)


def build_fluid_fraction_summary(
    admissible_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for skey, slabel in STAGE_LABELS.items():
        for metric, unit in [
            ("fluid_mass_kg_per_1kgAOC", "kg per 1 kg initial AOC"),
            ("fluid_fraction_of_initial_AOC", "fraction"),
            ("fluid_fraction_of_stage_start", "fraction"),
        ]:
            arr = finite_array(admissible_df[f"{skey}_{metric}"])
            rows.append({
                "stage": slabel,
                "stage_key": skey,
                "variable": metric,
                "unit": unit,
                **numeric_summary(arr),
            })

    return pd.DataFrame(rows)


def build_small_fluid_diagnostic(
    admissible_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for skey, slabel in STAGE_LABELS.items():
        arr = finite_array(
            admissible_df[f"{skey}_fluid_fraction_of_stage_start"]
        )
        for threshold in FLUID_FRACTION_DIAGNOSTIC_THRESHOLDS:
            rows.append({
                "stage": slabel,
                "stage_key": skey,
                "threshold_fraction": threshold,
                "N_below_threshold": int(np.sum(arr < threshold)),
                "fraction_below_threshold": (
                    float(np.mean(arr < threshold))
                    if len(arr) else np.nan
                ),
                "note": (
                    "Diagnostic only; these realizations are not automatically "
                    "deleted from the main model."
                ),
            })

    return pd.DataFrame(rows)


# =============================================================================
# 5. CONVERGENCE AND H2O-SAMPLING SENSITIVITY
# =============================================================================

def build_convergence_table(
    admissible_df: pd.DataFrame,
    n_total_attempted: int,
) -> pd.DataFrame:
    """
    Use path_id prefixes of the main run, so all convergence points are nested.
    """
    rows = []

    checkpoints = sorted(
        set(
            n for n in CONVERGENCE_ATTEMPTS
            if 0 < n <= n_total_attempted
        )
    )
    if n_total_attempted not in checkpoints:
        checkpoints.append(n_total_attempted)

    for n_attempt in checkpoints:
        sub = admissible_df[
            admissible_df["path_id"] <= n_attempt
        ].copy()

        n_good = len(sub)
        rows.append({
            "attempted_paths": n_attempt,
            "metric_scope": "H2O_path",
            "stage": "whole_path",
            "variable": "physically_admissible",
            "value": n_good / n_attempt if n_attempt else np.nan,
            "N_support": n_good,
        })

        if sub.empty:
            continue

        process = build_element_process_summary(sub)
        fluid_stats = build_conditional_fluid_summary(sub, process)

        for _, r in process.iterrows():
            rows.append({
                "attempted_paths": n_attempt,
                "metric_scope": "element_process",
                "stage": r["stage"],
                "variable": f"{r['element']}_P_net_loss",
                "value": r["P_net_loss_given_evaluable"],
                "N_support": r["N_element_evaluable"],
            })

        for _, r in fluid_stats.iterrows():
            rows.append({
                "attempted_paths": n_attempt,
                "metric_scope": "conditional_fluid",
                "stage": r["stage"],
                "variable": f"{r['element']}_median_{r['unit']}",
                "value": r["median"],
                "N_support": r["N"],
            })

    return pd.DataFrame(rows)


def summarize_mode_for_sensitivity(
    admissible_df: pd.DataFrame,
    inadmissible_df: pd.DataFrame,
    n_attempted: int,
    mode: str,
) -> pd.DataFrame:
    rows = []

    p_path = len(admissible_df) / n_attempted if n_attempted else np.nan
    rows.append({
        "H2O_sampling_mode": mode,
        "metric_scope": "H2O_path",
        "stage": "whole_path",
        "variable": "P_physically_admissible",
        "value": p_path,
        "N_support": len(admissible_df),
    })

    process = build_element_process_summary(admissible_df)
    fluid_stats = build_conditional_fluid_summary(admissible_df, process)

    for _, r in process.iterrows():
        rows.append({
            "H2O_sampling_mode": mode,
            "metric_scope": "element_process",
            "stage": r["stage"],
            "variable": f"{r['element']}_P_net_loss",
            "value": r["P_net_loss_given_evaluable"],
            "N_support": r["N_element_evaluable"],
        })

    for _, r in fluid_stats.iterrows():
        rows.append({
            "H2O_sampling_mode": mode,
            "metric_scope": "conditional_fluid",
            "stage": r["stage"],
            "variable": f"{r['element']}_median_{r['unit']}",
            "value": r["median"],
            "N_support": r["N"],
        })

    return pd.DataFrame(rows)


def run_h2o_sensitivity(
    df: pd.DataFrame,
    main_admissible: pd.DataFrame,
    main_inadmissible: pd.DataFrame,
) -> pd.DataFrame:
    tables = []

    for mode in H2O_SENSITIVITY_MODES:
        if mode == H2O_SAMPLING_MAIN and N_SENSITIVITY == N_SIM:
            good = main_admissible
            bad = main_inadmissible
        else:
            good, bad = run_simulation(
                df=df,
                n_sim=N_SENSITIVITY,
                seed=SENSITIVITY_SEED,
                h2o_mode=mode,
                verbose=True,
            )

        tables.append(
            summarize_mode_for_sensitivity(
                admissible_df=good,
                inadmissible_df=bad,
                n_attempted=N_SENSITIVITY,
                mode=mode,
            )
        )

    return pd.concat(tables, ignore_index=True)


# =============================================================================
# 6. FIGURES
# =============================================================================

def save_figure(fig: plt.Figure, output_base: Path) -> None:
    fig.savefig(
        output_base.with_suffix(".png"),
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )
    fig.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    fig.savefig(
        output_base.with_suffix(".svg"),
        bbox_inches="tight",
    )
    plt.close(fig)


def robust_display_limits(arr: np.ndarray) -> Tuple[float, float]:
    """
    Display-only P0.5-P99.5 range. Full results remain in output tables.
    """
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0, 1.0

    lo, hi = np.percentile(arr, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = np.min(arr), np.max(arr)
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def format_median_text(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.3g}"


def add_boxplot_histogram_with_stats(
    arrays,
    labels,
    colors,
    ylabel,
    title,
    output_base: Path,
    xscale_hist: str = "linear",
) -> None:
    """Create a boxplot with side histogram and per-box median/n annotations."""
    safe_arrays = [a if len(a) else np.array([np.nan], dtype=float) for a in arrays]
    fig = plt.figure(figsize=(10.4, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.10, 0.95), wspace=0.28)
    ax_box = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])

    bp = ax_box.boxplot(safe_arrays, tick_labels=labels, showfliers=False,
                        patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.60)
        patch.set_edgecolor("#444444")
    for median in bp["medians"]:
        median.set_color("#8B1E3F")
        median.set_linewidth(1.5)

    finite_values = np.concatenate([a[np.isfinite(a)] for a in arrays if len(a)]) if any(len(a) for a in arrays) else np.array([])
    if finite_values.size:
        y0, y1 = np.nanmin(finite_values), np.nanmax(finite_values)
        if np.isfinite(y0) and np.isfinite(y1) and y1 > y0:
            ypad = (y1 - y0) * 0.08
            ax_box.set_ylim(y0 - ypad, y1 + ypad)

    for i, arr in enumerate(arrays, start=1):
        finite_arr = arr[np.isfinite(arr)]
        if len(finite_arr) == 0:
            continue
        med = float(np.median(finite_arr))
        q3 = float(np.percentile(finite_arr, 75))
        text_y = q3
        if np.isfinite(text_y):
            ax_box.annotate(
                f"Median = {format_median_text(med)}\nn = {len(finite_arr)}",
                xy=(i, text_y),
                xytext=(14, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                          edgecolor=colors[i - 1], alpha=0.88, linewidth=0.8),
            )

    for arr, label, color in zip(arrays, labels, colors):
        finite_arr = arr[np.isfinite(arr)]
        if len(finite_arr) == 0:
            continue
        ax_hist.hist(finite_arr, bins=35, alpha=0.45, color=color,
                     edgecolor="white", linewidth=0.3,
                     label=f"{label} (n={len(finite_arr)})")
        med = float(np.median(finite_arr))
        ax_hist.axvline(med, color=color, linestyle="--", linewidth=1.0)

    if xscale_hist == "log":
        positive = finite_values[finite_values > 0]
        if positive.size:
            ax_hist.set_xscale("log")
    ax_hist.set_ylabel("Monte Carlo count")
    ax_hist.set_xlabel(ylabel.splitlines()[0])
    ax_hist.legend(frameon=False, fontsize=8, loc="best")
    ax_hist.tick_params(direction="in")

    ax_box.set_ylabel(ylabel)
    ax_box.set_xlabel("Metamorphic dehydration stage")
    ax_box.set_title(title)
    ax_box.tick_params(direction="in")
    save_figure(fig, output_base)


def plot_element_process_probabilities(
    process_df: pd.DataFrame,
    outdir: Path,
) -> None:
    figdir = outdir / "figures_process"
    figdir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(STAGE_LABELS))
    stage_order = list(STAGE_LABELS.values())

    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    for elem in ELEMENTS:
        sub = (
            process_df[process_df["element"] == elem]
            .set_index("stage")
            .reindex(stage_order)
        )
        ax.plot(
            x,
            sub["P_net_loss_given_evaluable"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.2,
            label=elem,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(stage_order)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(net loss | H$_2$O-valid, element evaluable)")
    ax.set_xlabel("Metamorphic dehydration stage")
    ax.legend(frameon=False, loc="best")
    ax.tick_params(direction="in")
    save_figure(fig, figdir / "element_net_loss_probability")


def plot_element_status_stacked(
    process_df: pd.DataFrame,
    outdir: Path,
) -> None:
    figdir = outdir / "figures_process"
    figdir.mkdir(parents=True, exist_ok=True)

    # One figure per element to keep interpretation transparent.
    for elem in ELEMENTS:
        sub = process_df[process_df["element"] == elem].copy()
        sub = sub.set_index("stage").reindex(list(STAGE_LABELS.values()))

        net = sub["P_net_loss_given_evaluable"].to_numpy(dtype=float)
        ret = sub[
            "P_near_zero_retention_given_evaluable"
        ].to_numpy(dtype=float)
        gain = sub["P_apparent_gain_given_evaluable"].to_numpy(dtype=float)

        x = np.arange(len(sub))
        fig, ax = plt.subplots(figsize=(6.8, 5.4))
        ax.bar(x, net, label="Net loss")
        ax.bar(x, ret, bottom=net, label="Near-zero retention")
        ax.bar(x, gain, bottom=net + ret, label="Apparent gain")

        ax.set_xticks(x)
        ax.set_xticklabels(sub.index.tolist())
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of element-evaluable realizations")
        ax.set_xlabel("Metamorphic dehydration stage")
        ax.legend(frameon=False, loc="best")
        ax.tick_params(direction="in")

        save_figure(
            fig,
            figdir / f"{elem}_process_classification_stacked",
        )


def plot_conditional_fluid_boxplots(
    admissible_df: pd.DataFrame,
    outdir: Path,
) -> None:
    figdir = outdir / "figures_conditional_fluid_boxplot"
    figdir.mkdir(parents=True, exist_ok=True)

    for elem, unit in ELEMENTS.items():
        arrays = []
        labels = []
        for skey, slabel in STAGE_LABELS.items():
            col = f"{skey}_{elem}_fluid_{unit}_conditional_net_loss"
            arrays.append(finite_array(admissible_df[col]))
            labels.append(slabel)

        if not any(len(a) for a in arrays):
            continue

        xscale_hist = "log" if all((len(a) == 0) or np.all(a > 0) for a in arrays) else "linear"
        add_boxplot_histogram_with_stats(
            arrays=arrays,
            labels=labels,
            colors=list(STAGE_COLORS.values()),
            ylabel=f"{elem} in released fluid ({unit})\nconditional on net loss",
            title=f"{elem} released-fluid distribution by stage",
            output_base=figdir / f"{elem}_conditional_net_loss_boxplot",
            xscale_hist=xscale_hist,
        )


def plot_conditional_fluid_violin(
    admissible_df: pd.DataFrame,
    outdir: Path,
) -> None:
    figdir = outdir / "figures_conditional_fluid_violin"
    figdir.mkdir(parents=True, exist_ok=True)

    for elem, unit in ELEMENTS.items():
        data = []
        positions = []
        labels = []

        for i, (skey, slabel) in enumerate(
            STAGE_LABELS.items(), start=1
        ):
            arr = finite_array(
                admissible_df[
                    f"{skey}_{elem}_fluid_{unit}_conditional_net_loss"
                ]
            )
            if len(arr):
                data.append(arr)
                positions.append(i)
                labels.append((i, slabel))

        if not data:
            continue

        fig, ax = plt.subplots(figsize=(6.8, 5.4))
        vp = ax.violinplot(
            data,
            positions=positions,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body, pos in zip(vp["bodies"], positions):
            body.set_facecolor(list(STAGE_COLORS.values())[pos - 1])
            body.set_edgecolor("#555555")
            body.set_alpha(0.60)
        ax.set_xticks([p for p, _ in labels])
        ax.set_xticklabels([lab for _, lab in labels])
        ax.set_ylabel(
            f"{elem} in released fluid ({unit})\n"
            "conditional on net loss"
        )
        ax.set_xlabel("Metamorphic dehydration stage")
        ax.tick_params(direction="in")
        save_figure(
            fig,
            figdir / f"{elem}_conditional_net_loss_violin",
        )


def plot_conditional_fluid_histograms(
    admissible_df: pd.DataFrame,
    outdir: Path,
) -> None:
    figdir = outdir / "figures_conditional_fluid_histogram"
    figdir.mkdir(parents=True, exist_ok=True)

    for skey, slabel in STAGE_LABELS.items():
        for elem, unit in ELEMENTS.items():
            arr = finite_array(
                admissible_df[
                    f"{skey}_{elem}_fluid_{unit}_conditional_net_loss"
                ]
            )
            if len(arr) == 0:
                continue

            lo, hi = robust_display_limits(arr)

            fig, ax = plt.subplots(figsize=(6.8, 5.4))
            ax.hist(arr, bins=60, range=(lo, hi),
                    color=STAGE_COLORS[skey], alpha=0.60, edgecolor="white",
                    linewidth=0.2)
            ax.axvline(
                np.median(arr),
                linestyle="--",
                linewidth=1.2,
            )
            ax.set_xlabel(
                f"{elem} in released fluid ({unit}), "
                "conditional on net loss"
            )
            ax.set_ylabel("Monte Carlo count")
            ax.set_title(slabel)
            ax.tick_params(direction="in")
            save_figure(
                fig,
                figdir / f"{skey}_{elem}_conditional_histogram",
            )


def plot_conditional_ratio_boxplots(
    admissible_df: pd.DataFrame,
    outdir: Path,
) -> None:
    figdir = outdir / "figures_conditional_ratio_boxplot"
    figdir.mkdir(parents=True, exist_ok=True)

    for num, den in RATIOS:
        arrays = []
        labels = []

        for skey, slabel in STAGE_LABELS.items():
            col = (
                f"{skey}_{num}_{den}_mass_ratio_"
                "conditional_joint_net_loss"
            )
            arrays.append(finite_array(admissible_df[col]))
            labels.append(slabel)

        if not any(len(a) for a in arrays):
            continue

        add_boxplot_histogram_with_stats(
            arrays=arrays,
            labels=labels,
            colors=list(STAGE_COLORS.values()),
            ylabel=f"{num}/{den} mass ratio\nconditional on joint net loss",
            title=f"{num}/{den} released-fluid ratio by stage",
            output_base=figdir / f"{num}_{den}_joint_net_loss_boxplot",
            xscale_hist="log",
        )


def plot_conditional_median_intervals(
    fluid_summary_df: pd.DataFrame,
    ratio_summary_df: pd.DataFrame,
    outdir: Path,
) -> None:
    figdir = outdir / "figures_summary"
    figdir.mkdir(parents=True, exist_ok=True)

    x = np.arange(1, 4)
    stage_order = list(STAGE_LABELS.values())

    for elem, unit in ELEMENTS.items():
        sub = fluid_summary_df[
            fluid_summary_df["element"] == elem
        ].set_index("stage").reindex(stage_order)

        med = sub["median"].to_numpy(dtype=float)
        p16 = sub["P16"].to_numpy(dtype=float)
        p84 = sub["P84"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(6.8, 5.4))
        ax.errorbar(
            x,
            med,
            yerr=np.vstack([med - p16, p84 - med]),
            fmt="o-",
            capsize=4,
            linewidth=1.2,
            markersize=5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(stage_order)
        ax.set_ylabel(
            f"{elem} in released fluid ({unit})\n"
            "conditional median and P16–P84"
        )
        ax.set_xlabel("Metamorphic dehydration stage")
        ax.tick_params(direction="in")
        save_figure(
            fig,
            figdir / f"{elem}_conditional_median_P16_P84",
        )

    for num, den in RATIOS:
        ratio_name = f"{num}/{den}"
        sub = ratio_summary_df[
            (ratio_summary_df["ratio"] == ratio_name)
            & (ratio_summary_df["ratio_type"] == "mass")
        ].set_index("stage").reindex(stage_order)

        med = sub["median"].to_numpy(dtype=float)
        p16 = sub["P16"].to_numpy(dtype=float)
        p84 = sub["P84"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(6.8, 5.4))
        ax.errorbar(
            x,
            med,
            yerr=np.vstack([med - p16, p84 - med]),
            fmt="o-",
            capsize=4,
            linewidth=1.2,
            markersize=5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(stage_order)
        ax.set_ylabel(
            f"{ratio_name} mass ratio\n"
            "conditional median and P16–P84"
        )
        ax.set_xlabel("Metamorphic dehydration stage")
        ax.tick_params(direction="in")
        save_figure(
            fig,
            figdir / f"{num}_{den}_conditional_median_P16_P84",
        )


def plot_convergence(
    convergence_df: pd.DataFrame,
    outdir: Path,
) -> None:
    if convergence_df.empty:
        return

    figdir = outdir / "figures_convergence"
    figdir.mkdir(parents=True, exist_ok=True)

    # One figure per element for P(net loss).
    for elem in ELEMENTS:
        fig, ax = plt.subplots(figsize=(6.8, 5.4))

        for stage in STAGE_LABELS.values():
            sub = convergence_df[
                (convergence_df["metric_scope"] == "element_process")
                & (convergence_df["stage"] == stage)
                & (convergence_df["variable"] == f"{elem}_P_net_loss")
            ].sort_values("attempted_paths")

            if len(sub):
                ax.plot(
                    sub["attempted_paths"],
                    sub["value"],
                    marker="o",
                    linewidth=1.2,
                    label=stage,
                )

        ax.set_ylim(0, 1)
        ax.set_xlabel("Attempted Monte Carlo paths")
        ax.set_ylabel(f"{elem}: P(net loss | evaluable)")
        ax.legend(frameon=False, loc="best")
        ax.tick_params(direction="in")
        save_figure(
            fig,
            figdir / f"{elem}_net_loss_probability_convergence",
        )


def plot_h2o_sensitivity(
    sensitivity_df: pd.DataFrame,
    outdir: Path,
) -> None:
    if sensitivity_df.empty:
        return

    figdir = outdir / "figures_H2O_sensitivity"
    figdir.mkdir(parents=True, exist_ok=True)

    modes = list(H2O_SENSITIVITY_MODES)
    stage_order = list(STAGE_LABELS.values())

    for elem in ELEMENTS:
        fig, ax = plt.subplots(figsize=(7.0, 5.4))
        x = np.arange(len(stage_order))

        for mode in modes:
            values = []
            for stage in stage_order:
                sub = sensitivity_df[
                    (sensitivity_df["H2O_sampling_mode"] == mode)
                    & (sensitivity_df["metric_scope"] == "element_process")
                    & (sensitivity_df["stage"] == stage)
                    & (sensitivity_df["variable"] == f"{elem}_P_net_loss")
                ]
                values.append(
                    float(sub["value"].iloc[0])
                    if len(sub) else np.nan
                )

            ax.plot(
                x,
                values,
                marker="o",
                linewidth=1.2,
                label=mode,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(stage_order)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"{elem}: P(net loss | evaluable)")
        ax.set_xlabel("Metamorphic dehydration stage")
        ax.legend(frameon=False, loc="best")
        ax.tick_params(direction="in")
        save_figure(
            fig,
            figdir / f"{elem}_H2O_sampling_sensitivity",
        )


# =============================================================================
# 6b. CONCENTRATION-VS-INVENTORY AUDIT + SIX-PANEL RELEASED-FLUID FIGURE
# =============================================================================

def make_f_reconciliation_audit(input_df: pd.DataFrame,
                                good: pd.DataFrame,
                                outdir: Path) -> pd.DataFrame:
    """Compare first-code-style mean concentration contrasts to the *third*
    code's paired rock-mass elemental-inventory changes. These are distinct
    estimands: unpaired population means vs paired simulated inventories.
    No gross F release can be inferred for net-gain paths without external
    uptake and gross-release constraints.
    """
    rows = []
    for skey, (up, down) in STAGE_PAIRS.items():
        sub_up = pd.to_numeric(input_df.loc[input_df["stage"] == up,
                                              "F_ppm"], errors="coerce").dropna()
        sub_down = pd.to_numeric(input_df.loc[input_df["stage"] == down,
                                                "F_ppm"], errors="coerce").dropna()
        mean_up = float(sub_up.mean()) if len(sub_up) else np.nan
        mean_down = float(sub_down.mean()) if len(sub_down) else np.nan
        aoc = pd.to_numeric(input_df.loc[input_df["stage"] == "AOC",
                                           "F_ppm"], errors="coerce").dropna()
        mean_aoc = float(aoc.mean()) if len(aoc) else np.nan
        first_pct = (100 * (mean_up - mean_down) / mean_aoc
                     if np.isfinite(mean_aoc) and mean_aoc > 0 else np.nan)
        status = good[f"{skey}_F_process_status"].astype(str)
        eval_mask = status.isin(["net_loss", "near_zero_retention", "apparent_gain"])
        n_eval = int(eval_mask.sum())
        diffs = pd.to_numeric(good[f"{skey}_F_concentration_difference_ppm"],
                              errors="coerce")
        losses = pd.to_numeric(good[f"{skey}_F_loss_mg_per_1kgAOC"],
                               errors="coerce")
        paired_mask = np.isfinite(diffs) & np.isfinite(losses) & eval_mask
        sign_discord = ((diffs[paired_mask] < 0) & (losses[paired_mask] > 0)) | (
                       (diffs[paired_mask] > 0) & (losses[paired_mask] < 0))
        rows.append({
            "stage": STAGE_LABELS[skey],
            "N_F_upstream_measured": len(sub_up),
            "N_F_downstream_measured": len(sub_down),
            "mean_upstream_F_ppm": mean_up,
            "mean_downstream_F_ppm": mean_down,
            "first_code_concentration_only_%_of_mean_AOC": first_pct,
            "N_H2O_valid_paths": len(good),
            "N_F_evaluable_paths": n_eval,
            "N_F_net_loss_paths": int((status == "net_loss").sum()),
            "N_F_apparent_gain_paths": int((status == "apparent_gain").sum()),
            "N_F_missing_paths": int((status == "missing").sum()),
            "P_F_net_loss_given_evaluable": (float((status == "net_loss").sum()/n_eval)
                                                if n_eval else np.nan),
            "P_F_apparent_gain_given_evaluable": (float((status == "apparent_gain").sum()/n_eval)
                                                    if n_eval else np.nan),
            "N_paired_concentration_inventory_opposite_sign": int(sign_discord.sum()),
            "note": ("Mean concentration gain is compatible with net release in "
                     "some sampled mass-adjusted paths; F fluid is defined only "
                     "on net-loss paths. Gross release on net-gain paths "
                     "requires independently constrained external uptake."),
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(outdir / "F_first_vs_third_mass_balance_audit.csv",
                 index=False, encoding="utf-8-sig")
    return audit


def build_released_fluid_plot_ratios(good: pd.DataFrame, outdir: Path):
    """Build fluid-only 2D plotting datasets; no mantle mixing or imputation.

    F/Cl versus I/Cl: F, Cl, I must all be net-loss in the SAME path.
    Br/Cl versus I/Cl: Br, Cl, I must all be net-loss in the SAME path.
    The two panels may use different path subsets (F or Br can be missing).
    All ratios are MASS ratios; Br and I are converted ppb -> ppm.
    This simultaneous-path filter affects *2D display only*; the elementwise
    fluid concentration and ratio distributions retain all their own valid paths.
    """
    panels = {}
    diagnostics = []
    for skey in STAGE_PAIRS:
        fields = {element:
                  f"{skey}_{element}_fluid_{ELEMENTS[element]}_conditional_net_loss"
                  for element in ELEMENTS}
        stage_panels = {}
        for vertical_element in ("F", "Br"):
            needed = (vertical_element, "Cl", "I")
            statuses = {e: good[f"{skey}_{e}_process_status"].to_numpy()
                        for e in needed}
            measured = np.ones(len(good), dtype=bool)
            joint_loss = np.ones(len(good), dtype=bool)
            for e in needed:
                measured &= np.isin(statuses[e],
                                    ["net_loss", "near_zero_retention", "apparent_gain"])
                joint_loss &= statuses[e] == "net_loss"
            columns = [fields[e] for e in needed]
            values = good[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            valid = joint_loss & np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
            vals = values[valid].copy()
            if vertical_element == "Br":
                vals[:, 0] /= 1000.0  # Br fluid ppb -> ppm
            vals[:, 2] /= 1000.0      # I fluid ppb -> ppm
            stage_plot = pd.DataFrame({
                "path_id": good.loc[valid, "path_id"].to_numpy(),
                "I/Cl": vals[:, 2] / vals[:, 1] if len(vals) else np.array([], float),
                f"{vertical_element}/Cl": vals[:, 0] / vals[:, 1]
                if len(vals) else np.array([], float),
            })
            vertical_ratio = f"{vertical_element}/Cl"
            stage_plot.to_csv(outdir / f"released_fluid_{skey}_{vertical_element}_plot_ratios.csv",
                              index=False, encoding="utf-8-sig")
            stage_panels[vertical_ratio] = stage_plot
            diagnostics.append({
                "stage": STAGE_LABELS[skey],
                "panel": f"{vertical_ratio} vs I/Cl",
                "N_H2O_valid_paths": len(good),
                "N_three_elements_evaluable": int(measured.sum()),
                "N_three_elements_joint_net_loss": len(stage_plot),
                "P_three_joint_net_loss_given_evaluable":
                    len(stage_plot) / int(measured.sum()) if measured.any() else np.nan,
                "N_F_net_loss_paths": int((good[f"{skey}_F_process_status"] == "net_loss").sum()),
                "note": "Only original released-fluid compositions; no mantle mixing. "
                        "2D plot uses three jointly net-loss elements; "
                        "element-specific and pair-ratio summaries use their own valid subsets.",
            })
        panels[skey] = stage_panels
    diagnostic = pd.DataFrame(diagnostics)
    diagnostic.to_csv(outdir / "released_fluid_sixpanel_selection_audit.csv",
                      index=False, encoding="utf-8-sig")
    return panels, diagnostic


def _clean_log_arrays(x: np.ndarray, y: np.ndarray, xlim, ylim):
    mask = (np.isfinite(x) & np.isfinite(y) & (x >= xlim[0]) & (x <= xlim[1])
            & (y >= ylim[0]) & (y <= ylim[1]))
    return x[mask], y[mask], int(mask.sum())


def _kde_levels_on_log_grid(ax, x, y, base_color, xlim, ylim, rng):
    """20260805-style KDE of log10 ratios; contour density as % of peak."""
    if len(x) < 6:
        return
    from matplotlib.colors import to_rgba
    from scipy.stats import gaussian_kde
    if len(x) > FLUID_KDE_MAX_POINTS:
        indices = rng.choice(len(x), size=FLUID_KDE_MAX_POINTS, replace=False)
        x, y = x[indices], y[indices]
    xy = np.vstack((np.log10(x), np.log10(y)))
    if np.linalg.matrix_rank(np.cov(xy)) < 2:
        return
    try:
        estimator = gaussian_kde(xy)
        gx = np.linspace(np.log10(xlim[0]), np.log10(xlim[1]), FLUID_KDE_GRID)
        gy = np.linspace(np.log10(ylim[0]), np.log10(ylim[1]), FLUID_KDE_GRID)
        X, Y = np.meshgrid(gx, gy)
        z = estimator(np.vstack((X.ravel(), Y.ravel()))).reshape(X.shape)
    except (ValueError, np.linalg.LinAlgError):
        return
    peak = float(np.max(z))
    if not np.isfinite(peak) or peak <= 0:
        return
    # Pale -> intense, same 20,40,60,80,100% fractions as second code.
    cutoff = np.array([.20, .40, .60, .80, 1.0]) * peak
    color = to_rgba(base_color)
    shades = [(*color[:3], alpha) for alpha in (.20, .40, .60, .80)]
    if np.min(z) < cutoff[0]:
        ax.contourf(10**X, 10**Y, z, levels=cutoff,
                    colors=shades, antialiased=True, zorder=2)
        ax.contour(10**X, 10**Y, z, levels=cutoff[:-1],
                   colors=[(*color[:3], .65)], linewidths=.42,
                   linestyles="dotted", zorder=3)


def _log_marginal(ax, values, limits, color, horizontal=False):
    from scipy.stats import gaussian_kde
    if len(values) == 0:
        return
    edges = np.logspace(np.log10(limits[0]), np.log10(limits[1]),
                        FLUID_HIST_BINS + 1)
    counts, _ = np.histogram(values, bins=edges)
    if horizontal:
        ax.barh(edges[:-1], counts, height=np.diff(edges), align="edge",
                color=color, alpha=.50, edgecolor="#555555", linewidth=.22)
        ax.set_ylim(*limits)
    else:
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
               color=color, alpha=.50, edgecolor="#555555", linewidth=.22)
        ax.set_xlim(*limits)
    logv = np.log10(values)
    if len(logv) > 4 and np.std(logv) > 1e-12:
        if len(logv) > FLUID_KDE_MAX_POINTS:
            # deterministic thinning keeps plotting cost bounded
            logv = logv[np.linspace(0, len(logv)-1, FLUID_KDE_MAX_POINTS).astype(int)]
        try:
            kde = gaussian_kde(logv)
            smooth = np.linspace(np.log10(limits[0]), np.log10(limits[1]), 220)
            log_bin_width = (np.log10(limits[1]) - np.log10(limits[0])) / FLUID_HIST_BINS
            curve = kde(smooth) * len(values) * log_bin_width
            if horizontal:
                ax.plot(curve, 10**smooth, color="#9D4D4D", lw=.85)
            else:
                ax.plot(10**smooth, curve, color="#9D4D4D", lw=.85)
        except (ValueError, np.linalg.LinAlgError):
            pass
    ax.tick_params(direction="in", length=2, width=.65,
                   labelsize=7, pad=1.5)
    if horizontal:
        ax.yaxis.set_ticks_position("right")
        ax.tick_params(left=False, labelleft=False, right=False,
                       labelright=False)
        ax.xaxis.set_ticks_position("top")
        ax.tick_params(bottom=False, labelbottom=False, top=True,
                       labeltop=True)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    else:
        ax.yaxis.set_ticks_position("right")
        ax.tick_params(left=False, labelleft=False, right=True,
                       labelright=True)
        ax.tick_params(bottom=False, labelbottom=False, top=False,
                       labeltop=False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)


def _expand_log_bounds(reference, values):
    """Retain reference bounds but include all valid points when requested."""
    if not AUTO_EXPAND_FLUID_AXES:
        return reference
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return reference
    floor_value = min(reference[0], float(values.min()))
    ceil_value = max(reference[1], float(values.max()))
    return (float(10**np.floor(np.log10(floor_value))),
            float(10**np.ceil(np.log10(ceil_value))))


def plot_six_panel_released_fluids(good: pd.DataFrame, outdir: Path):
    """3 x 2 reference-style layout for ORIGINAL released-fluid ratios.

    Rows: AOC->BS, BS->HP, HP->UHP.
    Columns: F/Cl vs I/Cl and Br/Cl vs I/Cl.
    Top/right axes: marginal histogram and KDE of the same plotted paths.
    No DMM endmember, mantle fractions, mixed fluids, or observational overlays.
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import LogLocator, LogFormatterMathtext

    panels, diagnostic = build_released_fluid_plot_ratios(good, outdir)
    # Keep a common x scale for the six panels to preserve comparability.
    pooled_x = [panel["I/Cl"].to_numpy(dtype=float)
                for stage in panels.values() for panel in stage.values()]
    xlim = _expand_log_bounds(FLUID_PLOT_XLIM,
                              np.concatenate(pooled_x) if pooled_x else np.array([]))
    with plt.rc_context({"font.family": "Times New Roman", "font.size": 9,
                         "axes.linewidth": .8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none"}):
        fig = plt.figure(figsize=(11.1, 15.1), facecolor="white")
        outer = GridSpec(3, 2, figure=fig, left=.078, right=.965,
                         bottom=.049, top=.984, wspace=.23, hspace=.32)
        rng = np.random.default_rng(20260920)
        for row, skey in enumerate(STAGE_PAIRS):
            rgb = STAGE_COLORS[skey]
            for col, (yname, ylim_ref) in enumerate((("F/Cl", FLUID_PLOT_FYLIM),
                                                     ("Br/Cl", FLUID_PLOT_BRYLIM))):
                panel = panels[skey][yname]
                ylim = _expand_log_bounds(ylim_ref, panel[yname].to_numpy(dtype=float))
                inner = outer[row, col].subgridspec(
                    2, 2, width_ratios=(1, .120), height_ratios=(.118, 1),
                    hspace=.014, wspace=.017)
                ax = fig.add_subplot(inner[1, 0])
                top = fig.add_subplot(inner[0, 0], sharex=ax)
                right = fig.add_subplot(inner[1, 1], sharey=ax)
                corner = fig.add_subplot(inner[0, 1])
                corner.axis("off")
                ax.set(xscale="log", yscale="log", xlim=xlim,
                       ylim=ylim, xlabel="I/Cl", ylabel=yname)
                ax.tick_params(which="major", length=4.5, width=.75,
                               direction="in", labelsize=9)
                ax.tick_params(which="minor", length=2.5, width=.65,
                               direction="in")
                ax.xaxis.set_major_locator(LogLocator(base=10))
                ax.yaxis.set_major_locator(LogLocator(base=10))
                ax.xaxis.set_major_formatter(LogFormatterMathtext())
                ax.yaxis.set_major_formatter(LogFormatterMathtext())
                ax.text(.5, .977, f"Released fluid ({row+1})",
                        transform=ax.transAxes, va="top", ha="center",
                        fontsize=10.8, zorder=11)
                ax.text(.032, .968, "ABCDEF"[row*2+col],
                        transform=ax.transAxes, va="top", ha="left",
                        fontsize=12, fontweight="bold", zorder=11)
                ax.text(.50, .905, STAGE_LABELS[skey],
                        transform=ax.transAxes, va="top", ha="center",
                        fontsize=8, color="#555555", zorder=11)
                x, y, n_display = _clean_log_arrays(
                    panel["I/Cl"].to_numpy(dtype=float),
                    panel[yname].to_numpy(dtype=float), xlim, ylim)
                top.set_xscale("log")
                right.set_yscale("log")
                if len(x):
                    _kde_levels_on_log_grid(ax, x, y, rgb, xlim, ylim, rng)
                    _log_marginal(top, x, xlim, rgb)
                    _log_marginal(right, y, ylim, rgb, horizontal=True)
                else:
                    ax.text(.50, .50, "No jointly released F/Cl/I or Br/Cl/I data",
                            transform=ax.transAxes, ha="center", va="center",
                            fontsize=7.5, color="#555555", wrap=True)
                top.set_xlim(xlim)
                right.set_ylim(ylim)
                top.tick_params(axis="x", bottom=False, labelbottom=False)
                right.tick_params(axis="y", left=False, labelleft=False)
                if col == 0:
                    inset = ax.inset_axes([.035, .045, .46, .022], zorder=12)
                    inset.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto",
                                 cmap=LinearSegmentedColormap.from_list(
                                     "density_"+skey, ["#FFFFFF", rgb]))
                    inset.set_xticks([0, 51, 102, 153, 204, 255],
                                     labels=["0", "20%", "40%", "60%", "80%", "100%"],
                                     fontsize=5.1)
                    inset.set_yticks([])
                    inset.set_title("Density (fraction of max)", fontsize=5.5, pad=0)
                    inset.tick_params(length=0, pad=0)
                print(f"{skey}, {yname}: {len(panel):,} jointly net-loss fluid paths; "
                      f"{n_display:,} displayed")
        for extension in ("svg", "pdf"):
            fig.savefig(outdir / f"Fig_fluid_composition_six_panels.{extension}",
                        bbox_inches="tight")
        fig.savefig(outdir / "Fig_fluid_composition_six_panels.jpg",
                    bbox_inches="tight", dpi=PLOT_DPI)
        plt.close(fig)
    return diagnostic


# =============================================================================
# 7. EXCEL OUTPUT
# =============================================================================

def autosize_excel_columns(writer: pd.ExcelWriter) -> None:
    for ws in writer.book.worksheets:
        for col_cells in ws.columns:
            max_len = 0
            col_letter = col_cells[0].column_letter
            for cell in col_cells[:5000]:
                try:
                    value = "" if cell.value is None else str(cell.value)
                    max_len = max(max_len, len(value))
                except Exception:
                    pass
            ws.column_dimensions[col_letter].width = min(
                max(max_len + 2, 10),
                45,
            )
        ws.freeze_panes = "A2"


def write_excel(
    outdir: Path,
    input_df: pd.DataFrame,
    input_audit_df: pd.DataFrame,
    h2o_path_df: pd.DataFrame,
    element_process_df: pd.DataFrame,
    conditional_fluid_df: pd.DataFrame,
    ratio_process_df: pd.DataFrame,
    conditional_ratio_df: pd.DataFrame,
    fluid_fraction_df: pd.DataFrame,
    small_fluid_df: pd.DataFrame,
    convergence_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    admissible_df: pd.DataFrame,
) -> Path:
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "\nopenpyxl is required for Excel output.\n"
            "Install it in PowerShell with:\n"
            r"C:\Users\A\AppData\Local\Programs\Python\Python39\python.exe "
            "-m pip install openpyxl\n"
        ) from exc

    path = outdir / "MC_halogen_NC_summary.xlsx"

    audit_n = min(10_000, len(admissible_df))
    audit_df = admissible_df.head(audit_n).copy()

    config_df = pd.DataFrame([
        ["N_SIM", N_SIM],
        ["RANDOM_SEED", RANDOM_SEED],
        ["CHUNK_SIZE", CHUNK_SIZE],
        ["H2O_SAMPLING_MAIN", H2O_SAMPLING_MAIN],
        ["ELEMENT_CHANGE_REL_TOL", ELEMENT_CHANGE_REL_TOL],
        ["RUN_CONVERGENCE", RUN_CONVERGENCE],
        ["RUN_H2O_SENSITIVITY", RUN_H2O_SENSITIVITY],
        ["N_SENSITIVITY", N_SENSITIVITY],
        [
            "model",
            "path-constrained progressive open-system whole-rock mass balance",
        ],
        [
            "H2O_constraint",
            "AOC >= BS >= HP >= UHP",
        ],
        [
            "mass_basis",
            "initial AOC = 1 kg; anhydrous mass conserved along full path",
        ],
        [
            "external_input_assumption",
            "no external fluid or solid input in the imposed mass-balance model",
        ],
        [
            "missing_halogen",
            "element-wise NaN; no interpolation, mean filling or zero filling",
        ],
        [
            "element_reporting",
            "net_loss / near_zero_retention / apparent_gain / missing / no_fluid_release",
        ],
        [
            "fluid_reporting",
            "fluid concentration is conditional on element net loss",
        ],
        [
            "ratio_reporting",
            "ratio distribution is conditional on joint net loss",
        ],
        [
            "six_panel_plot",
            "released fluid only; no mantle mixing; 2D plotting subsets require three joint net losses",
        ],
    ], columns=["parameter", "value"])

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        input_df.to_excel(
            writer, sheet_name="Input_data", index=False
        )
        input_audit_df.to_excel(
            writer, sheet_name="Input_audit", index=False
        )
        config_df.to_excel(
            writer, sheet_name="Model_config", index=False
        )
        h2o_path_df.to_excel(
            writer, sheet_name="H2O_path_summary", index=False
        )
        element_process_df.to_excel(
            writer, sheet_name="Element_process", index=False
        )
        conditional_fluid_df.to_excel(
            writer, sheet_name="Conditional_fluid", index=False
        )
        ratio_process_df.to_excel(
            writer, sheet_name="Ratio_process", index=False
        )
        conditional_ratio_df.to_excel(
            writer, sheet_name="Conditional_ratio", index=False
        )
        fluid_fraction_df.to_excel(
            writer, sheet_name="Fluid_fraction", index=False
        )
        small_fluid_df.to_excel(
            writer, sheet_name="Small_fluid_diag", index=False
        )

        if not convergence_df.empty:
            convergence_df.to_excel(
                writer, sheet_name="Convergence", index=False
            )
        if not sensitivity_df.empty:
            sensitivity_df.to_excel(
                writer, sheet_name="H2O_sensitivity", index=False
            )

        audit_df.to_excel(
            writer, sheet_name="MC_audit_first10000", index=False
        )

        autosize_excel_columns(writer)

    return path


# =============================================================================
# 8. MAIN
# =============================================================================

def main() -> None:
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print("NC-oriented path-constrained Monte Carlo halogen dehydration model")
    print("=" * 92)

    df = load_and_validate_input(INPUT_CSV)

    print("\nInput sample counts:")
    print(df.groupby("stage").size().reindex(VALID_STAGES))

    input_audit_df = build_input_audit(df)

    # -------------------------------------------------------------------------
    # Main simulation
    # -------------------------------------------------------------------------
    admissible_df, inadmissible_df = run_simulation(
        df=df,
        n_sim=N_SIM,
        seed=RANDOM_SEED,
        h2o_mode=H2O_SAMPLING_MAIN,
        verbose=True,
    )

    if admissible_df.empty:
        raise RuntimeError(
            "No physically admissible H2O paths were generated. "
            "Check the stage-specific H2O ranges."
        )

    h2o_path_df = build_h2o_path_summary(
        admissible_df=admissible_df,
        inadmissible_df=inadmissible_df,
        n_attempted=N_SIM,
        mode=H2O_SAMPLING_MAIN,
    )
    element_process_df = build_element_process_summary(admissible_df)
    conditional_fluid_df = build_conditional_fluid_summary(
        admissible_df,
        element_process_df,
    )
    ratio_process_df = build_ratio_process_summary(admissible_df)
    conditional_ratio_df = build_conditional_ratio_summary(
        admissible_df,
        ratio_process_df,
    )
    fluid_fraction_df = build_fluid_fraction_summary(admissible_df)
    small_fluid_df = build_small_fluid_diagnostic(admissible_df)

    # -------------------------------------------------------------------------
    # Convergence
    # -------------------------------------------------------------------------
    if RUN_CONVERGENCE:
        convergence_df = build_convergence_table(
            admissible_df=admissible_df,
            n_total_attempted=N_SIM,
        )
    else:
        convergence_df = pd.DataFrame()

    # -------------------------------------------------------------------------
    # H2O sensitivity
    # -------------------------------------------------------------------------
    if RUN_H2O_SENSITIVITY:
        sensitivity_df = run_h2o_sensitivity(
            df=df,
            main_admissible=admissible_df,
            main_inadmissible=inadmissible_df,
        )
    else:
        sensitivity_df = pd.DataFrame()

    # -------------------------------------------------------------------------
    # CSV outputs
    # -------------------------------------------------------------------------
    input_audit_df.to_csv(
        outdir / "input_data_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    h2o_path_df.to_csv(
        outdir / "H2O_path_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    element_process_df.to_csv(
        outdir / "element_process_probabilities.csv",
        index=False,
        encoding="utf-8-sig",
    )
    conditional_fluid_df.to_csv(
        outdir / "conditional_fluid_composition_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ratio_process_df.to_csv(
        outdir / "ratio_joint_net_loss_probabilities.csv",
        index=False,
        encoding="utf-8-sig",
    )
    conditional_ratio_df.to_csv(
        outdir / "conditional_halogen_ratio_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fluid_fraction_df.to_csv(
        outdir / "fluid_fraction_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    small_fluid_df.to_csv(
        outdir / "small_fluid_fraction_diagnostic.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not convergence_df.empty:
        convergence_df.to_csv(
            outdir / "monte_carlo_convergence.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if not sensitivity_df.empty:
        sensitivity_df.to_csv(
            outdir / "H2O_sampling_sensitivity.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if SAVE_ALL_H2O_VALID_PATHS:
        admissible_df.to_csv(
            outdir / "MC_physically_admissible_H2O_paths_all_results.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if SAVE_H2O_INADMISSIBLE_PATHS and len(inadmissible_df):
        inadmissible_df.to_csv(
            outdir / "MC_physically_inadmissible_H2O_paths.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # -------------------------------------------------------------------------
    # Excel
    # -------------------------------------------------------------------------
    excel_path = write_excel(
        outdir=outdir,
        input_df=df,
        input_audit_df=input_audit_df,
        h2o_path_df=h2o_path_df,
        element_process_df=element_process_df,
        conditional_fluid_df=conditional_fluid_df,
        ratio_process_df=ratio_process_df,
        conditional_ratio_df=conditional_ratio_df,
        fluid_fraction_df=fluid_fraction_df,
        small_fluid_df=small_fluid_df,
        convergence_df=convergence_df,
        sensitivity_df=sensitivity_df,
        admissible_df=admissible_df,
    )

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------
    plot_element_process_probabilities(
        element_process_df,
        outdir,
    )
    plot_element_status_stacked(
        element_process_df,
        outdir,
    )
    plot_conditional_fluid_boxplots(
        admissible_df,
        outdir,
    )
    plot_conditional_fluid_violin(
        admissible_df,
        outdir,
    )
    plot_conditional_fluid_histograms(
        admissible_df,
        outdir,
    )
    plot_conditional_ratio_boxplots(
        admissible_df,
        outdir,
    )
    plot_conditional_median_intervals(
        conditional_fluid_df,
        conditional_ratio_df,
        outdir,
    )

    if not convergence_df.empty:
        plot_convergence(
            convergence_df,
            outdir,
        )
    if not sensitivity_df.empty:
        plot_h2o_sensitivity(
            sensitivity_df,
            outdir,
        )

    # -------------------------------------------------------------------------
    # First-code vs third-code F sign diagnostic and fluid-only 3 x 2 figure
    # -------------------------------------------------------------------------
    f_audit_df = make_f_reconciliation_audit(df, admissible_df, outdir)
    print("\nF concentration-vs-inventory reconciliation:")
    print(f_audit_df[["stage", "first_code_concentration_only_%_of_mean_AOC",
                      "P_F_net_loss_given_evaluable",
                      "P_F_apparent_gain_given_evaluable"]].to_string(index=False))
    if MAKE_FLUID_6PANEL_FIGURE:
        fluid_plot_audit = plot_six_panel_released_fluids(admissible_df, outdir)
        print("\nReleased-fluid plot selection (three elements jointly net-loss):")
        print(fluid_plot_audit[["stage", "panel",
                                "N_three_elements_joint_net_loss",
                                "P_three_joint_net_loss_given_evaluable"]].to_string(index=False))

    # -------------------------------------------------------------------------
    # Mass-balance diagnostics
    # -------------------------------------------------------------------------
    max_fluid_resid = np.nanmax(
        np.abs(admissible_df["fluid_mass_balance_residual_kg"])
    )

    print("\nMass-balance diagnostics:")
    print(
        f"Max absolute fluid-mass residual = "
        f"{max_fluid_resid:.3e} kg"
    )

    for elem in ELEMENTS:
        arr = finite_array(
            admissible_df[f"{elem}_mass_balance_residual_mg"]
        )
        if len(arr):
            print(
                f"Max absolute {elem} telescoping residual = "
                f"{np.max(np.abs(arr)):.3e} mg"
            )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    print("\nH2O-path summary:")
    print(h2o_path_df.to_string(index=False))

    print("\nElement process probabilities:")
    display = element_process_df[
        [
            "stage",
            "element",
            "N_element_evaluable",
            "N_net_loss",
            "N_near_zero_retention",
            "N_apparent_gain",
            "N_missing",
            "P_net_loss_given_evaluable",
        ]
    ]
    print(display.to_string(index=False))

    print("\nIMPORTANT:")
    print(
        "Fluid-composition statistics are conditional on net-loss "
        "realizations; ratio statistics are conditional on joint-net-loss "
        "realizations."
    )

    print(f"\nExcel summary: {excel_path.resolve()}")
    print(f"All outputs:   {outdir.resolve()}")
    print("=" * 92)


if __name__ == "__main__":
    main()
