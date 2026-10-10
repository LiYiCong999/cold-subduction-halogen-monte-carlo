# Monte Carlo models for halogen behavior during cold oceanic subduction

Reproducibility repository containing three Python scripts supporting analyses of fluorine (F), chlorine (Cl), bromine (Br), and iodine (I) behavior during progressive cold oceanic subduction. The scripts are documented in the scientific presentation order: **release efficiency → halogen dehydration → mélange mixing**. They represent separate calculations and should **not** be interpreted as a single coupled numerical model or software-dependent pipeline.

## Contents

| Script | Scientific aim | Input |
|---|---|---|
| `scripts/01_stage_release_efficiency.py` | Element-specific bootstrap estimates of AOC-normalized stage-by-stage halogen release contributions | `data/release_efficiency.csv` |
| `scripts/02_open_system_halogen_dehydration.py` | Stage-resolved open-system dehydration (AOC → blueschist → HP eclogite → UHP eclogite), sampling whole-rock halogen and sample-specific water bounds; conditional fluid halogen compositions | `data/bulk_data.csv` |
| `scripts/03_melange_mixing_mc.py` | Three-endmember Monte Carlo mélange mixing (metasediment, metamafic rock, forearc serpentinite), with three metamafic facies and six ratio panels | `data/halogen_data.csv` |

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

**Mélange mixing (script 03).** Three endmembers are sampled and mixed under Dirichlet-distributed proportions, with blueschist, HP-eclogite and UHP-eclogite metamafic scenarios treated separately. The plot compares F/Cl and Br/Cl versus I/Cl under the script's original assumptions.

## Provenance and modifications

Three user-provided Python scripts were incorporated into this repository. Their numerical methods, filters, random seeds, stage ordering, and plotting settings were retained. Local Windows input and output paths were adapted to project-relative paths, and Chinese comments, docstrings, and console messages in script 01 were translated into English. Original filenames and SHA-256 checksums are recorded in [`docs/SOURCE_PROVENANCE.json`](docs/SOURCE_PROVENANCE.json). The adjusted scripts passed Python static syntax checks. The three input CSV datasets are now included in this repository. Full numerical execution and reproducibility of the Monte Carlo simulations have not yet been independently verified.

## Citation, release and data availability

1. **Author:** Yi-Cong Li. Software citation metadata is provided in [`CITATION.cff`](CITATION.cff).
2. **Source code and input data:** The three Python scripts and their corresponding CSV datasets are available in the `scripts/` and `data/` directories.
3. **Release and DOI:** The initial release (`v1.0.0`) has been archived on Zenodo. The version-specific DOI is [10.5281/zenodo.23278295](https://doi.org/10.5281/zenodo.23278295). This DOI identifies the original v1.0.0 release and does not include subsequent GitHub modifications.

4. **Licenses:** The three Python scripts in [`scripts/`](scripts/) are provided under the [MIT License](scripts/LICENSE). The three input CSV datasets in [`data/`](data/) are provided under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](data/LICENSE). The original Zenodo v1.0.0 archive predates these license-file additions and displays CC BY 4.0 in its record metadata.

See [`docs/GITHUB_ZENODO_GUIDE_CN.md`](docs/GITHUB_ZENODO_GUIDE_CN.md) for detailed Chinese instructions.
