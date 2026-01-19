# Adventures in demand analysis using AI: Reproducibility Package

This repository contains the code, configurations, and output folders used for the JASA submission on Adventures in demand analysis using AI. 

## Project scope
- **Main analysis**: notebooks in `main/code` reproduce the paper’s primary empirical results (embedding construction, clustering, predictive performance, evaluation).
- **Appendix analyses**: notebooks in `appendix/code` reproduce supplemental experiments (substitute prices, dimensionality reduction via neural nets, replication on clothes data).
- **Outputs**: generated figures/tables are written to `output/` folders under each track (main and appendix).

## Data availability
- Prediction/embedding files referenced in `paths_config.yaml` are stored in this repository under `main/data/predictions` and `appendix/data/**`. These files are derived from proprietary product data.
- Some intermediate datasets hosted on Huggingface used during the analysis are anonymized in the code.
- No raw product data are redistributed here. If you need access during review, please contact the authors.

## Computational environment
- Language: Python (tested with 3.11.11).
- Install dependencies via the provided `requirements.txt` inside each `code` directory (run `pip install -r requirements.txt` from that folder after activating your virtual environment).
- Hardware: 
    * Processor: 13th Gen Intel(R) Core(TM) i7-13620H, 1700 Mhz, 10 Core(s), 16 Logical Processor(s)
    * RAM: 16 GB
    * OS Name: Microsoft Windows 11 Home

## How to reproduce the main results
1) **Use provided datasets**: The processed datasets are already in `main/data`. You can skip the dataset creation notebooks unless you need to regenerate intermediates.
2) **Run notebooks in order** (from `main/code`):
	- `01_1_create_dataset_txt_img.ipynb` (optional)
	- `01_2_create_dataset_txt.ipynb` (optional)
	- `02_cluster_centroid_products.ipynb`
	- `03_1_predictive_performance_txt_img.ipynb`
	- `03_2_predictive_performance_txt.ipynb`
	- `04_evaluation.ipynb`
3) **Check outputs**: notebooks write results to `main/output`, including `04_evaluation/sorted_effects.csv`.

## How to reproduce appendix results
Follow the instructions in `appendix/README.md`. Each appendix module has its own notebook sequence and output folder.

## Contact
For questions or review-time data access, please reach out to the corresponding author listed in the manuscript.

