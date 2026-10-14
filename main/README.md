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

## Variant: EmbeddingGemma 2 instead of fine-tuned embeddings
Re-runs `04_evaluation.ipynb` with off-the-shelf [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) embeddings (`google/embeddinggemma-2`, revision `914f7f8`) instead of the fine-tuned (DoubleMLDeep) embeddings. All embedding-derived variables (`emb_*`, `pca_*`, `similarity_cluster_*`, nearest-neighbour substitutes) are recomputed with the same functions and seeds as in `01_1`; all other variables and the estimation sample are identical to the main analysis.

Environment: Python 3.11 with `code/requirements_gemma.txt`.

1. `code/01_3_compute_embeddings_gemma.ipynb`: one 768-d embedding per product (the product text and image are time-invariant), stored in `data/embeddings_gemma/`.
   - `txtimg`: product text and image interleaved into one multimodal embedding (`"task: clustering | query: {text} <|image|>"`). `txt`: text only.
   - bfloat16 inference, default vision budget (280 tokens per image). On a 4-core CPU this takes about 0.2 s per product for `txt` and 2.5 s for `txtimg`.
2. `code/01_4_create_dataset_gemma.ipynb`: builds `data/dataset_gemma_{txtimg,txt}_dim{256,768}_{train,val}.zip`. Matryoshka truncation is followed by L2 re-normalization.
3. `code/04_evaluation_gemma.ipynb`: copy of `04_evaluation.ipynb` with a parameters cell and an export of the key results.
   - The committed outputs are the main specification: text+image, 256 dimensions, the same as the fine-tuned embeddings.
   - The other runs are stored as executed notebooks in `output/04_evaluation_gemma/<run>/`: `embeddinggemma2_txt_dim256`, `embeddinggemma2_txtimg_dim768`, and `finetuned`. `finetuned` is the original specification re-run in the same environment, and it reproduces `04_evaluation.ipynb` exactly.
4. `code/04_evaluation_gemma_comparison.ipynb`: side-by-side comparison, written to `output/04_evaluation_gemma/comparison/`.

Main results (90% CIs):

| | Fine-tuned (txt+img, 256d) | Gemma 2 (txt+img, 256d) | Gemma 2 (txt, 256d) | Gemma 2 (txt+img, 768d) |
| --- | --- | --- | --- | --- |
| Linear, lagged + embeddings | -0.657 [-0.722, -0.592] | -0.681 [-0.746, -0.616] | -0.679 [-0.745, -0.614] | -0.666 [-0.732, -0.600] |
| Linear, lagged + controls + embeddings | -0.712 [-0.776, -0.649] | -0.731 [-0.795, -0.667] | -0.729 [-0.793, -0.664] | -0.715 [-0.780, -0.651] |
| PLR, embeddings | -0.623 [-0.704, -0.542] | -0.652 [-0.725, -0.580] | -0.586 [-0.668, -0.505] | -0.612 [-0.683, -0.541] |
| PLR, embeddings + controls | -0.697 [-0.777, -0.617] | -0.701 [-0.776, -0.625] | -0.636 [-0.714, -0.557] | -0.706 [-0.778, -0.634] |
| PLR, similarities + controls | -0.691 [-0.757, -0.624] | -0.684 [-0.749, -0.620] | -0.677 [-0.742, -0.612] | -0.697 [-0.764, -0.630] |
| R² outcome / treatment (PLR, emb. + controls) | 0.891 / 0.974 | 0.895 / 0.976 | 0.898 / 0.974 | 0.894 / 0.976 |
| Robustness value RV (PLR, emb. + controls) | 0.149 | 0.147 | 0.141 | 0.149 |
| χ² p-value, heterogeneity along clusters (PLR) | 0.0009 | 0.0116 | 0.0162 | 0.0066 |
