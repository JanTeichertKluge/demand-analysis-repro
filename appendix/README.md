# Appendix Analyses (Supplementary Results)

This folder hosts the supplemental experiments reported in the appendix.

## A2: Substitute Prices
- Notebook: `code/A2_substitute_prices/01_evaluation_substitute_prices.ipynb`.
- Purpose: evaluate substitute-price construction using nearest-neighbor embeddings.
- Inputs: prediction/embedding CSVs from the `main/data` directory (if you relocate them, update the paths inside the notebook before execution).
- Output: tables saved under `output/A2_substitute_prices`.

## A3: Dimensionality Reduction via Neural Nets
- Notebooks: `code/A3_dimensionality_reduction_via_NN/01_create_dataset.ipynb`, `02_evaluation.ipynb`.
- Purpose: build reduced-dimensional embeddings and assess predictive quality.
- Inputs: derived CSVs placed according to notebook instructions (use `appendix/data/A3_dimensionality_reduction_via_NN`).
- Output: evaluation results under `output/A3_dimensionality_reduction_via_NN`.

## A4: Replication on Clothes Data
- Notebooks: `code/A4_replication_clothes_data/01_create_dataset.ipynb`, `02_cluster_centroid_products.ipynb`, `03_predictive_performance.ipynb`, `04_evaluation.ipynb`.
- Utilities: `code/A4_replication_clothes_data/utils/paths_config.yaml`, `utils_data2.py`, `utils_models.py`.
- Execution order matches the numbering above, run each notebook top-to-bottom.
- Inputs: prediction/embedding CSVs referenced in the A4 `paths_config.yaml` file (update paths if your files are elsewhere).
- Outputs: generated under `output/A4_replication_clothes_data` (cluster results in `01_output_cluster`, evaluation tables in `02_output_evaluation`).

## Environment
- Language: Python (tested with 3.11.11).
- Install dependencies via the provided `requirements.txt` inside each `code` directory.