# Main Analysis (Paper Results)

This folder reproduces the primary results reported in the paper. All steps are implemented as Jupyter notebooks and rely on the prediction/embedding CSVs referenced in `code/utils/paths_config.yaml`.

## Inputs
- Prediction and embedding CSVs for text-only and text+image models. Relative paths are configured in `code/utils/paths_config.yaml`. Update these paths if you store the files elsewhere.
- No raw product data are redistributed here; only derived prediction/embedding files are needed.

## Environment
- Python 3.10+ with pandas, numpy, scikit-learn, statsmodels, linearmodels, doubleml, scipy, pyyaml, jupyter.
- Optional: set `PYTHONHASHSEED` and scikit-learn random states to keep clustering results identical.

## Execution order
Run the notebooks in this order after ensuring the CSVs exist at the configured paths:
1. `code/01_create_dataset.ipynb`
2. `code/02_cluster_centroid_products.ipynb`
3. `code/03_1_predictive_performance_txt_img.ipynb`
4. `code/03_2_predictive_performance_txt.ipynb`
5. `code/04_evaluation.ipynb`

Execute each notebook top-to-bottom. The utilities in `code/utils` are imported by the notebooks; no additional scripts are required.

## Outputs
- Intermediate artifacts are written under `output/` (e.g., clusters, prediction diagnostics).
- Final evaluation tables, including `output/04_evaluation/sorted_effects.csv`, are produced by `04_evaluation.ipynb`.

## Notes
- If you relocate prediction files, update `code/utils/paths_config.yaml` before running.
- All notebooks assume relative paths are resolved from their own directory.
- For deterministic clustering/PCA results, keep scikit-learn seeds unchanged.
