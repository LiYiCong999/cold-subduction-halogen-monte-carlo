# Input datasets

This directory contains the three CSV input datasets used for Monte Carlo simulations of halogen behavior during cold oceanic subduction.

## 1. Release efficiency

**File:** `release_efficiency.csv`

**Script:** `scripts/01_stage_release_efficiency.py`

- Required columns: `Rock_Type`, `F_ppm`, `Cl_ppm`, and either `Br_ppm`, `I_ppm` or `Br_ppb`, `I_ppb`.
- Rock types: AOC, Blueschist, HP eclogite, and UHP eclogite.
- Purpose: Estimation of stage-dependent halogen release contributions.

## 2. Halogen dehydration

**File:** `bulk_data.csv`

**Script:** `scripts/02_open_system_halogen_dehydration.py`

- Required columns: `stage`, `sample_id`, `H2O_min_wt%`, `H2O_max_wt%`, `F_ppm`, `Cl_ppm`, `Br_ppb`, `I_ppb`.
- Stages: AOC, BS, HP, and UHP.
- Purpose: Monte Carlo modeling of progressive open-system dehydration and released-fluid halogen compositions.

## 3. Mélange mixing

**File:** `halogen_data.csv`

**Script:** `scripts/03_melange_mixing_mc.py`

- Required columns: `Rock_Type`, `F_ppm`, `Cl_ppm`, `Br_ppb`, `I_ppb`.
- Rock types: metasediment, forearc serpentinite, Blueschist, HP eclogite, and UHP eclogite.
- Purpose: Three-endmember Monte Carlo simulation of mélange halogen compositions.

## Data availability

All three input CSV datasets are included in this GitHub repository and are intended to be archived with the corresponding software release on Zenodo.

The source code, input datasets, and documentation are provided to support reproducibility of the Monte Carlo calculations.

## Units

- F and Cl concentrations: ppm.
- Br and I concentrations: ppb in scripts 02 and 03.
- Script 01 accepts Br and I in ppm or ppb and converts units when necessary.
- Water contents: wt%.
