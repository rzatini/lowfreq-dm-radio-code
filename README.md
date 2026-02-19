# lowfreq-dm-radio-code — Reproducible Plots for the Paper

This repository contains the notebooks and shared modules used to reproduce the figures in the paper. The workflow is organized by target (Earth, Jupiter, Sun), followed by profile and sensitivity plots.

## Repository structure (root)
- `01_Earth.ipynb` — Earth target pipeline (profiles → probabilities/fluxes).
- `02_Jupiter.ipynb` — Jupiter target pipeline.
- `03_Sun.ipynb` — Sun target pipeline.
- `04_Plot_Profiles.ipynb` — target profile plots (e.g., plasma frequency, gradients, B‑fields).
- `05_Plot_Sensitivities.ipynb` — final sensitivity curves and comparison plots.
- `modules/` — shared physics models, constants, conversions, and noise models.
- `data/` — instrument data, bounds, trajectories, and auxiliary inputs.
- `grids/` — saved flux grids and derived outputs used by the plotting notebooks.

## Purpose
Reproduce the plots used in the paper by running the notebooks in order (targets → profiles → sensitivities), using a consistent set of shared models.

## Python version
- Python **3.12.8** (from notebook kernel metadata)

## Suggested usage order
1) Run `01_Earth.ipynb`, `02_Jupiter.ipynb`, `03_Sun.ipynb` to generate flux grids.
2) Run `04_Plot_Profiles.ipynb` to generate target profile plots.
3) Run `05_Plot_Sensitivities.ipynb` for final sensitivity curves.

## Data provenance
See `data/README.md` for detailed data sources, digitization notes, and bounds references.

## Notes / good practice
- Keep output grids in `grids/` so notebooks load from a single location.
- If plots use LaTeX rendering, ensure a working TeX installation is available.
