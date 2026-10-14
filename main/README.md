# Main Analysis (Paper Results)

This folder reproduces the primary results reported in the paper. All steps are implemented as Jupyter notebooks and rely on the prediction/embedding CSVs referenced in `code/utils/paths_config.yaml`.

## Inputs
- Prediction and embedding CSVs for text-only and text+image models. Relative paths are configured in `code/utils/paths_config.yaml`. Update these paths if you store the files elsewhere.
- No raw product data are redistributed here, only processed data.

## Environment
- Python 3.11.11 with dependencies listed in `code/requirements.txt`. 

## Execution order
Run the notebooks in this order after ensuring the CSVs exist at the configured paths:
1. `code/01_1_create_dataset_txt_img.ipynb` (optional, regenerates text+image datasets)
2. `code/01_2_create_dataset_txt.ipynb` (optional, regenerates text-only datasets)
3. `code/02_cluster_centroid_products.ipynb`
4. `code/03_1_predictive_performance_txt_img.ipynb`
5. `code/03_2_predictive_performance_txt.ipynb`
6. `code/04_evaluation.ipynb`

## Outputs
- Intermediate artifacts as well as final results are written under `output/` (e.g., clusters, prediction diagnostics).

## Notes
- If you relocate prediction files, update `code/utils/paths_config.yaml` before running.
- All notebooks assume relative paths are resolved from their own directory.
- For deterministic clustering/PCA results, keep scikit-learn seeds unchanged.
