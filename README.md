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

The study input datasets were not attached when this software package was prepared. Place the actual CSV files in `data/`, using the names shown in the table above. See [`data/README.md`](data/README.md) for expected columns and concentration units. **Without these data, the scripts cannot reproduce the published results.**

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

Three user-provided Python files were incorporated. Their numerical methods, filters, random seeds, stage ordering and plot settings were left unchanged. **Only three local Windows input paths and three output paths were changed** to portable project-relative locations; Chinese source comments, docstrings, and human-facing console/error messages in script 01 were translated to English. Scripts 02 and 03 already contained no Chinese text. The SHA-256 checksums of the original uploads and their original filenames are recorded in [`docs/SOURCE_PROVENANCE.json`](docs/SOURCE_PROVENANCE.json). These adjusted scripts passed Python static syntax compilation; scientific calculations were **not** rerun, because their input CSV datasets were not supplied.

## Citation, release and data availability

1. Complete [`CITATION.cff.template`](CITATION.cff.template) with the real code-author name(s), then save it as `CITATION.cff` **before** making a tagged GitHub Release. Do not release with placeholder names.
2. Connect the repository to Zenodo, enable repository archiving, then create a GitHub release (for example `v1.0.0`). Zenodo normally archives each eligible release and assigns a version-specific DOI.
3. Replace this section with the permanent repository URL, version DOI, and any separate input-data DOI once available.
4. Decide and add a license with the actual rights holders' approval; no license is assigned in this preparation package.

See [`docs/GITHUB_ZENODO_GUIDE_CN.md`](docs/GITHUB_ZENODO_GUIDE_CN.md) for detailed Chinese instructions.
