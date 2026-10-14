# EmbeddingGemma 2 Variant of the Main Analysis

This folder re-runs the main analysis (`main/code`, notebooks 01 to 04) with off-the-shelf [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) embeddings in place of the fine-tuned (DoubleMLDeep) product embeddings. The notebooks have the same names as in `main/code`. Each notebook begins with a note listing what differs from its `main` counterpart.

All embedding-derived variables are recomputed from the Gemma embeddings with the same functions and seeds as in `main/code/01_1` and `01_2`: `emb_*`, `pca_*`, `similarity_cluster_*`, and the nearest-neighbour substitute prices. All other variables and the estimation sample are identical to the main analysis, which `01_1` and `01_2` check with assertions.

## Embeddings
- Model `google/embeddinggemma-2`, revision `914f7f8`, loaded with sentence-transformers.
- One embedding per product (product texts and images are time-invariant), computed with the prompt `task: clustering | query: `.
- `txtimg`: the product text and image are interleaved into one multimodal embedding (`{text} <|image|>`, 280 vision tokens per image). `txt`: text only.
- bfloat16 inference. On a 4-core CPU this takes about 0.2 s per product for `txt` and 2.5 s for `txtimg`.
- Matryoshka truncation to 256 dimensions (the dimension of the fine-tuned embeddings), followed by L2 re-normalization.
- Where the main analysis uses text-only embeddings (`01_2`, `03_2`, and the text part of `02_cluster_centroid_products`), the Gemma `txt` embeddings are used. Everywhere else the `txtimg` embeddings are used.

## Structure
- `code/`: notebooks, `utils/utils_gemma.py` (loading the model, embedding, and caching), and `requirements.txt`. The other utils are copies of `main/code/utils`.
- `data/embeddings/`: Gemma embeddings (768 dimensions, Parquet) and their metadata.
- `data/dataset_txt_only_{False,True}_embeddings_True_{train,val}.zip`: the datasets of `main/data` with the embedding-derived columns replaced. The `_dim768` files hold the 768-dimensional `txtimg` variant.
- `outputs/`: one folder per notebook, as in `main/output`.
- Raw product data and the original datasets are read from `main/data`.

## Environment
Python 3.11 with `code/requirements.txt`. These are the main requirements plus torch, torchvision, transformers, sentence-transformers, pyarrow, and kaleido (static export of the plotly figures). The model (Apache 2.0) is downloaded from Hugging Face on first use.

## Execution order
Run from `code/`:
1. `00_compute_embeddings.ipynb`: computes and caches the embeddings (`txt` and `txtimg`).
2. `01_1_create_dataset_txt_img.ipynb` and `01_2_create_dataset_txt.ipynb`: build the datasets.
3. `02_cluster_centroid_products.ipynb`: clusters, 3D plots, and products closest to the centroids. It also contains the cluster headlines. These were generated with Claude, because Gemini 3 Pro (used in `main`) was not available, and the GenAI images of `main` were not reproduced.
4. `03_1_predictive_performance_txt_img.ipynb` and `03_2_predictive_performance_txt.ipynb`. The deep-learning rows (`Deep Time (In)dependent`) are the fine-tuned models of the main analysis and serve as a benchmark.
5. `04_evaluation.ipynb`: main specification (`txtimg`, 256 dimensions). It has a parameters cell (`data_dir`, `txt_only`, `emb_dim`, `output_dir`) and exports the key results as CSV and JSON.
6. `02_cluster_summary.ipynb`: requires `outputs/04_evaluation/sorted_effects.csv`.
7. `04_evaluation_comparison.ipynb`: compares four runs of `04_evaluation.ipynb`. Three are the extra runs below, and one is the main specification. The extra runs are stored as executed notebooks in `outputs/04_evaluation_comparison/<run>/`:
   - `finetuned`: the original specification (`data_dir="../../main/data"`), re-run in this environment. It reproduces `main/code/04_evaluation.ipynb` exactly.
   - `gemma_txt_dim256`: text-only embeddings (`txt_only=True`).
   - `gemma_txtimg_dim768`: full 768 dimensions (`emb_dim=768`).

`04_evaluation.ipynb` corresponds to `main/code/04_evaluation.ipynb` as of commit `f352b25`. It therefore contains the grid of sensitivity scenarios, not the calibrated scenarios and the main elasticity table that were added to `main` later.

## Main results
Inverse-rank elasticities with 90% CIs (`outputs/04_evaluation_comparison/ate_estimates.csv`):

| | Fine-tuned (txt+img, 256d) | Gemma 2 (txt+img, 256d) | Gemma 2 (txt, 256d) | Gemma 2 (txt+img, 768d) |
| --- | --- | --- | --- | --- |
| Linear, lagged + embeddings | -0.657 [-0.722, -0.592] | -0.681 [-0.746, -0.616] | -0.679 [-0.745, -0.614] | -0.666 [-0.732, -0.600] |
| Linear, lagged + controls + embeddings | -0.712 [-0.776, -0.649] | -0.731 [-0.795, -0.667] | -0.729 [-0.793, -0.664] | -0.715 [-0.780, -0.651] |
| Linear, lagged + controls + similarities | -0.723 [-0.786, -0.659] | -0.735 [-0.799, -0.670] | -0.734 [-0.798, -0.669] | -0.734 [-0.798, -0.670] |
| Linear with interactions | -0.727 [-0.791, -0.663] | -0.736 [-0.801, -0.671] | -0.736 [-0.801, -0.671] | -0.737 [-0.802, -0.672] |
| PLR, embeddings | -0.623 [-0.704, -0.542] | -0.652 [-0.725, -0.580] | -0.586 [-0.668, -0.505] | -0.612 [-0.683, -0.541] |
| PLR, embeddings + controls | -0.697 [-0.777, -0.617] | -0.701 [-0.776, -0.625] | -0.636 [-0.714, -0.557] | -0.706 [-0.778, -0.634] |
| PLR, similarities + controls | -0.691 [-0.757, -0.624] | -0.684 [-0.749, -0.620] | -0.677 [-0.742, -0.612] | -0.697 [-0.764, -0.630] |
| PLR, PCA + controls | -0.677 [-0.749, -0.605] | -0.689 [-0.757, -0.622] | -0.690 [-0.757, -0.622] | -0.694 [-0.763, -0.625] |
| R² outcome / treatment (PLR, emb. + controls) | 0.891 / 0.974 | 0.895 / 0.976 | 0.898 / 0.974 | 0.894 / 0.976 |
| Robustness value RV (PLR, emb. + controls) | 0.149 | 0.147 | 0.141 | 0.149 |
| χ² p-value, heterogeneity along clusters (PLR) | 0.0009 | 0.0116 | 0.0162 | 0.0066 |

The cluster-similarity variables cannot be compared one-to-one across runs, because k-means is re-estimated for every embedding.

Predictive performance (test R², `03_1`, levels):

| Features | Q, fine-tuned | Q, Gemma 2 | P, fine-tuned | P, Gemma 2 |
| --- | --- | --- | --- | --- |
| Boosting (Tabular) | 0.475 | 0.475 | 0.185 | 0.185 |
| Boosting (Tabular + PCA) | 0.532 | 0.546 | 0.644 | 0.376 |
| Boosting (Tabular + Similarities) | 0.526 | 0.532 | 0.625 | 0.322 |
| Boosting (Tabular + Embeddings) | 0.553 | 0.588 | 0.656 | 0.555 |
| Deep, time dependent (fine-tuned) | 0.595 | – | 0.667 | – |

The fine-tuned embeddings were trained to predict prices and sales ranks. Their five principal components and cluster similarities therefore carry much more price information than the corresponding off-the-shelf Gemma features. With the full 256 dimensions, Gemma predicts quantities better and prices worse than the fine-tuned embeddings. The estimated elasticities are close in all specifications that combine the embeddings with the additional controls.
