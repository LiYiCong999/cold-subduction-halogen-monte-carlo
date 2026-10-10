# Required input data (not included in this package)

To reproduce the calculations, place the **actual, authorized study input datasets** in this folder:

- `release_efficiency.csv`: for `01_stage_release_efficiency.py`
  - Required `Rock_Type` and `F_ppm,Cl_ppm`, plus Br and I as either `Br_ppm,I_ppm` **or** `Br_ppb,I_ppb`.
  - The script expects `AOC`, `Blueschist`, `HP eclogite`, `UHP eclogite`.

- `bulk_data.csv`: for `02_open_system_halogen_dehydration.py`
  - Required columns: `stage,sample_id,H2O_min_wt%,H2O_max_wt%,F_ppm,Cl_ppm,Br_ppb,I_ppb`
  - `stage` must identify AOC, BS, HP, UHP (aliases handled in script).
  - Missing halogen measurements may be blank; the script applies its own conditional-evaluation handling.
- `halogen_data.csv`: for `03_melange_mixing_mc.py`
  - Required columns: `Rock_Type,F_ppm,Cl_ppm,Br_ppb,I_ppb`
  - `Rock_Type` must match: `metasediment`, `forearc serpentinite`, `Blueschist`, `HP eclogite`, or `UHP eclogite`.
These three CSVs were **not provided** in the materials used to prepare this repository, so they are not included. Do not publish unpublished or third-party data without permission. If data are published separately on Zenodo, provide the dataset DOI in the repository README and in the software record's related identifiers.

**Units:** F and Cl use ppm; Br and I use ppb in scripts 02 and 03, whereas script 01 converts ppb columns to ppm when necessary.
