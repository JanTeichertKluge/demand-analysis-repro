# Adventures in demand analysis using AI: Reproducibility Package

This repository contains the code, configurations, and output folders used for the JASA submission on Adventures in demand analysis using AI. 

## Project scope
- **Main analysis**: notebooks in `main/code` reproduce the paper’s primary empirical results (embedding construction, clustering, predictive performance, evaluation).
- **Appendix analyses**: notebooks in `appendix/code` reproduce supplemental experiments (substitute prices, dimensionality reduction via neural nets, replication on clothes data).
- **Outputs**: generated figures/tables are written to `output/` folders under each track (main and appendix).

## Data availability
- Input prediction/embedding CSVs referenced in `paths_config.yaml` are **not stored in this repository** because they are derived from proprietary product data. But the output data after preprocessing are included and are sufficient to reproduce all results.
- No raw data are redistributed here. If you need access during review, please contact the authors.

## Computational environment
- Language: Python (tested with 3.10+).
- Key packages: pandas, numpy, scikit-learn, statsmodels, linearmodels, doubleml, scipy, pyyaml, jupyter.
- Hardware: standard workstation; clustering and PCA steps benefit from ≥16 GB RAM.
- Set a virtual environment (e.g., `python -m venv .venv` then `pip install -U pip` and install the packages above).

## How to reproduce the main results
1) **Prepare data**: Populate the prediction/embedding CSVs referenced in `main/code/utils/paths_config.yaml` (see `main/README.md` for details).
2) **Run notebooks in order** (from `main/code`):
	- `01_create_dataset.ipynb`
	- `02_cluster_centroid_products.ipynb`
	- `03_1_predictive_performance_txt_img.ipynb`
	- `03_2_predictive_performance_txt.ipynb`
	- `04_evaluation.ipynb`
3) **Check outputs**: notebooks write results to `main/output`, including `04_evaluation/sorted_effects.csv`.

## How to reproduce appendix results
Follow the instructions in `appendix/README.md`. Each appendix module has its own notebook sequence and output folder.

## Repository layout (high level)
- `main/` — primary analysis notebooks, utilities, outputs.
- `appendix/` — supplemental analyses (A2–A4) with their own utilities and outputs.
- `manuscript/` — placeholder for paper materials.
- `output/` — top-level output aggregation (mirrors main results).

## ACC-style summary (readers’ view)
- **Data**: Proprietary prediction/embedding CSVs required; not bundled. Contact authors for review-time access.
- **Code**: All analysis code/notebooks are included in this repository.
- **Instructions**: Execution order and expected outputs documented in this README and subdirectory READMEs.
- **Computing environment**: Python 3.10+ with the packages listed above; no GPUs required.
- **Reproducibility status**: Deterministic given the supplied CSV inputs and fixed seeds in scikit-learn; reruns should match reported outputs.

## Contact
For questions or review-time data access, please reach out to the corresponding author listed in the manuscript.

