# Monte Carlo models for halogen behavior during cold oceanic subduction

Reproducibility repository containing three Python scripts supporting analyses of fluorine (F), chlorine (Cl), bromine (Br), and iodine (I) behavior during progressive cold oceanic subduction. The scripts are documented in the scientific presentation order: **release efficiency → halogen dehydration → mélange mixing**. They represent separate calculations and should **not** be interpreted as a single coupled numerical model or software-dependent pipeline.

## Contents

| Script | Scientific aim | Input |
|---|---|---|
| `scripts/01_stage_release_efficiency.py` | Element-specific bootstrap estimates of AOC-normalized stage-by-stage halogen release contributions | `data/release_efficiency.csv` |
| `scripts/02_open_system_halogen_dehydration.py` | Stage-resolved open-system dehydration (AOC → blueschist → HP eclogite → UHP eclogite), sampling whole-rock halogen and sample-specific water bounds; conditional fluid halogen compositions | `data/bulk_data.csv` |
| `scripts/03_melange_mixing_mc.py` |Three-endmember Monte Carlo simulations of mélange halogen compositions involving metasediments, metamafic rocks, and forearc or sub-arc serpentinites, with separate mixing scenarios and halogen-ratio visualizations | `data/halogen_data.csv` |

## System requirements

Python 3.10 or later, with NumPy, pandas, Matplotlib, SciPy and openpyxl (used by script 02 for Excel output). Package version ranges are in `requirements.txt`; exact original environment versions were **not supplied**.

### Installation (from the repository root)

```bash
python -m pip install -r requirements.txt
```

### Input files

The three input CSV datasets are included in the [`data/`](data/) directory: `release_efficiency.csv`, `bulk_data.csv`, and `halogen_data.csv`. See [`data/README.md`](data/README.md) for required columns, rock-type labels, and concentration units.

### Run (from the repository root)

```bash
python scripts/01_stage_release_efficiency.py
python scripts/02_open_system_halogen_dehydration.py
python scripts/03_melange_mixing_mc.py
```

Each script writes outputs under `results/` (git-ignored), and some scripts display figures interactively. Run one script at a time. Script 02 includes optional convergence and water-sampling sensitivity diagnostics. A fixed random seed is defined within each script.

## Calculation notes

**Relative stage contributions (script 01).** Bootstrap population means are drawn **independently for each element and rock type**. Relative stage contributions are normalized to the AOC reference abundance; negative values represent net apparent enrichment in this concentration-based diagnostic and should not automatically be interpreted as direct element fluxes.

**Dehydration (script 02).** Each realization samples one composition at each metamorphic stage. Sample-specific water contents are sampled from their corresponding input bounds. Only paths satisfying monotonic water loss are carried through to a stage-by-stage mass balance with conserved dry-rock mass and immediate removal of released fluid. A fluid halogen concentration is defined conditionally upon **net elemental loss**; halogen ratios require joint net loss. This is a statistical conceptual trajectory sampled from natural populations, not direct tracking of one continuous natural slab.

**Mélange mixing (script 03).** The revised model evaluates three-endmember mixing involving metasediments, metamafic rocks, and serpentinites, considering forearc and sub-arc serpentinite scenarios separately. The Monte Carlo simulations are used to characterize the variability of mélange halogen compositions and ratios under the prescribed mixing assumptions. These results represent statistical mixing scenarios rather than unique natural mixing histories.

## Provenance and modifications

The repository was initially developed from three user-provided Python scripts, with adjustments to file paths and documentation for public distribution. The original filenames and SHA-256 checksums are recorded in `docs/SOURCE_PROVENANCE.json`.

Compared with the archived v1.0.1 release, version 1.0.2 updates `scripts/02_open_system_halogen_dehydration.py` and `scripts/03_melange_mixing_mc.py`, together with the input dataset `data/halogen_data.csv`. These revisions incorporate updated dehydration visualizations and separate forearc and sub-arc serpentinite scenarios in the mélange mixing simulations. The stage-release efficiency script and the other two input CSV datasets remain unchanged.

The original source-file checksums document the initial uploads and do not represent the revised scripts. Full numerical reproducibility of the updated simulations has not yet been independently verified.

## Citation, release and data availability

1. **Author:** Yi-Cong Li. Software citation metadata is provided in [`CITATION.cff`](CITATION.cff).
2. **Source code and input data:** The three Python scripts and their corresponding CSV datasets are available in the `scripts/` and `data/` directories.
3. **Release and DOI:** Version `v1.0.3` has been archived on Zenodo with the version-specific DOI [10.5281/zenodo.24001068](https://doi.org/10.5281/zenodo.24001068). This release updates `scripts/02_open_system_halogen_dehydration.py`. The other two Python scripts and all three input CSV datasets remain unchanged from `v1.0.2`. The previous version `v1.0.2` remains archived at [10.5281/zenodo.23831719](https://doi.org/10.5281/zenodo.23831719).

4. **Licenses:** The three Python scripts in [`scripts/`](scripts/) are distributed under the [MIT License](scripts/LICENSE). The three input CSV datasets in [`data/`](data/) are distributed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](data/LICENSE), except for third-party materials subject to separate licensing terms. The software and dataset licenses are declared separately in the Zenodo record.

See [`docs/GITHUB_ZENODO_GUIDE_CN.md`](docs/GITHUB_ZENODO_GUIDE_CN.md) for detailed Chinese instructions.
