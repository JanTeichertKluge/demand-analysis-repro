# Fine-Tuned EmbeddingGemma 2 Variant of the Main Analysis

This folder repeats the EmbeddingGemma 2 variant (`gemma/`) with an EmbeddingGemma 2 model that is first fine-tuned on our data with LoRA, in the same way as the DoubleMLDeep model whose embeddings are used in `main/`.

## Fine-tuning
- Model `google/embeddinggemma-2`, revision `914f7f8`, prompt `task: clustering | query: `, as in `gemma/`.
- Sample split of the DoubleMLDeep model: the products of the HF `train` split ($I_1$, 3,774 products, 41,945 product-period rows) are used for the gradient updates; the epoch is selected and the learning rate reduced on the loss over all rows of the `validation` split ($I_2$, 3,775 products, 41,832 rows).
- Loss of the DoubleMLDeep model: $\mathrm{RMSE}(Q)\cdot\mathrm{RMSE}(P)$ over the product-period rows of a mini-batch, with $Q$ = `SALES_RANK` ($-\log$ sales rank) and $P$ = `BUYBOX_PRICE` (log price). Inputs are time-invariant, so each product is embedded once per step and the row-level loss is computed exactly from per-product statistics.
- Heads as in the DoubleMLDeep model: LayerNorm, then $\mathrm{Linear}(256,128)\to\mathrm{Linear}(128,1)$ per target, on the Matryoshka truncation of the embedding to 256 dimensions (the representation used downstream). The last layer starts at zero and its bias at the $I_1$ target means, so training starts from the constant predictor.
- LoRA ($r=16$, $\alpha=32$, dropout 0.05) on the attention and MLP projections of the language model and, for text + image, of the vision encoder. AdamW (learning rate $10^{-4}$ for LoRA, $10^{-3}$ for the heads), ReduceLROnPlateau (factor 0.1, patience 1) on the $I_2$ loss, early stopping after 4 epochs without an improvement of at least $10^{-4}$, at most 20 epochs. The text + image run reached the 20-epoch limit while the $I_2$ loss was still decreasing slightly (0.2971 to 0.2966 in epoch 20, selected epoch 20); the text-only run stopped early after epoch 19 (selected epoch 19).
- Differences from the DoubleMLDeep training: LoRA adapters instead of training the full network; mini-batches of 16 products with all their product-period rows (about 178 rows) instead of shuffled row-level mini-batches; AdamW instead of SGD with momentum (standard for LoRA), and therefore a learning-rate patience of 1 instead of 0 (with AdamW the epoch-level $I_2$ loss fluctuates enough that patience 0 cuts the learning rate on noise); the zero-initialized last layer described above; at most 20 epochs (the selected DoubleMLDeep checkpoints are from epoch 20, text only, and epoch 24, text + image; file names `epoch=019` and `epoch=023`, counted from 0). The other settings (learning rates, weight decay 0.01, gradient-norm clipping at 1.0, bfloat16, early stopping) are our own choices. The configuration of the DoubleMLDeep runs is not part of the repository; the DoubleMLDeep settings named here (SGD with learning rate 0.01 and momentum 0.99, ReduceLROnPlateau with factor 0.1 and patience 0, shuffled row-level batches) are the defaults of the `doubleml-deep` package and are assumed for these runs.
- Two runs: `txtimg` (text + image) and `txt` (text only, used where `main` uses text-only embeddings).

Reference values of the $I_2$ loss: constant predictor 0.818; DoubleMLDeep 0.332 (text + image) and 0.340 (text only); off-the-shelf EmbeddingGemma 2 at 256 dimensions with a linear ridge head on the $I_1$ product means about 0.48 (text + image) and 0.50 (text only) (computed outside the repository, for orientation).

## Running the fine-tuning (GPU)
- Colab: `code/00_finetune_gemma_colab.ipynb` (A100 or L4 runtime, Colab secret `GITHUB_TOKEN`, *Run all*). Checkpoints are kept on Google Drive and a run resumes after a disconnect. The delivered runs were made with this notebook on A100 GPUs (bfloat16): text + image 4.4 h for 20 epochs (resumed three times), text only 1.0 h for 19 epochs, about 5.4 h for both runs. An L4 needs considerably longer (not timed).
- Slurm: `code/slurm/finetune_gemma.sbatch` (see its header).
- Command line, from `code/`: `python finetune_gemma.py --modality txtimg --run-dir RUN_DIR` (and `--modality txt`). `--smoke` runs a short end-to-end test.

The code is in `code/finetune_gemma.py` and `code/utils/utils_finetune.py`. The outputs are written to `data/finetuned/<modality>/`: 768-dimensional L2-normalized embeddings of all products (`embeddinggemma2_tuned_<modality>.parquet`, with metadata `.json`), the predictions of the two heads for every product-period row (`predictions_<modality>.parquet`), the training history and the configuration. The LoRA adapters and the heads (`adapter_<modality>/`, `head_<modality>.pt`) are not part of the repository; they are only needed to recompute the embeddings.

Fine-tuning results (selected epoch; loss and $R^2$ of the time-invariant predictions of the heads, both computed over all 41,832 product-period rows of $I_2$). On the estimation sample of `03_1`/`03_2` (38,041 rows of $I_2$ after `dropna`), the same heads give $R^2$ 0.500/0.742 (text + image) and 0.493/0.704 (text only); these are the values in the Results table below:

| | $I_2$ loss | $R^2$ Q | $R^2$ P |
| --- | --- | --- | --- |
| text + image | 0.2966 (DoubleMLDeep: 0.3322) | 0.488 | 0.743 |
| text only | 0.3191 (DoubleMLDeep: 0.3401) | 0.483 | 0.705 |

## Pipeline
The notebooks in `code/` are generated from those in `gemma/code` (same names; each notebook begins with a note on what differs) with the fine-tuned embeddings in place of the off-the-shelf ones: all embedding-derived variables (`emb_*`, `pca_*`, `similarity_cluster_*`, nearest-neighbour substitute prices) are recomputed from `data/finetuned/` with the same functions and seeds as in `main/code`; all other variables and the estimation samples are identical to the main analysis (checked with assertions). Run from `code/`:
1. `01_1_create_dataset_txt_img.ipynb`, `01_2_create_dataset_txt.ipynb`: datasets in `data/` (text + image with 256 and 768 dimensions, text only with 256 dimensions).
2. `02_cluster_centroid_products.ipynb`: clusters, 3D plots and products closest to the centroids, with the cluster headlines (generated with Claude from the 4 products closest to the centroids of the `similarity_cluster_*` clusters, with the prompt of `main`). As in `main`, the clusters estimated in this notebook (3D plots, `collage_cluster_img_*`) are numbered differently from the `similarity_cluster_*` clusters used in the estimation (cross-tabulation at the end of the notebook); `collage_similarity_cluster_*` and the headlines use the numbering of the estimation.
3. `03_1_predictive_performance_txt_img.ipynb`, `03_2_predictive_performance_txt.ipynb`, with an additional row for the heads of the fine-tuned model.
4. `04_evaluation.ipynb`: main specification (text + image, 256 dimensions); additional runs with `txt_only=True` and `emb_dim=768` in `outputs/04_evaluation_comparison/<run>/`.
5. `02_cluster_summary.ipynb`: cluster table (as Table 9 of the paper), `outputs/R2/cluster_summary_table.csv`.
6. `04_evaluation_comparison.ipynb`: DoubleMLDeep (main analysis), off-the-shelf EmbeddingGemma 2 (`gemma/`) and the three runs of this folder.
7. `05_appendix_gemma_tuned.ipynb`: Section 7 of the Online Appendix, `outputs/05_appendix_gemma_tuned/Appendix_Gemma.tex`, from `templates/Appendix_Gemma.tex`: the text of Section 7.1 (off-the-shelf EmbeddingGemma 2) as in the paper, its tables extended by the fine-tuned EmbeddingGemma 2, and Section 7.2 (fine-tuned EmbeddingGemma 2), whose statements are checked against the estimates. The two figures of Section 7.2 in `outputs/05_appendix_gemma_tuned/figures/` are those of `04_evaluation_comparison.ipynb` with the legend labels of the Online Appendix.

Runtime on a 4-core CPU: about 2 h in total.

## Results
Predictive performance, $R^2$ (%) on the test sample ($I_2$), text + image (`03_1`); in parentheses text only (`03_2`):

| | DoubleMLDeep (`main`) | EmbeddingGemma 2 (`gemma`) | fine-tuned EmbeddingGemma 2 |
| --- | --- | --- | --- |
| Q, OLS (tabular + 5 PCs) | 46.5 (44.9) | 31.0 (30.0) | 48.2 (44.4) |
| P, OLS (tabular + 5 PCs) | 60.6 (55.0) | 16.5 (16.8) | 70.3 (66.1) |
| Q, boosting (tabular + similarities) | 52.6 (52.0) | 53.2 (54.9) | 52.1 (52.7) |
| P, boosting (tabular + similarities) | 62.5 (60.6) | 32.2 (34.6) | 71.8 (68.0) |
| Q, deep model (time independent) | 50.4 (50.3) | | 50.0 (49.3) |
| P, deep model (time independent) | 67.3 (65.7) | | 74.2 (70.4) |

The deep-model rows are the DoubleMLDeep models and the heads of the fine-tuned model. Fine-tuning recovers the price information that the off-the-shelf embeddings lack: a few principal components or cluster similarities of the fine-tuned embeddings again carry it, more than those of the DoubleMLDeep embeddings. For Q, fine-tuning adds little: the heads and the 5 principal components are on par with DoubleMLDeep, and with boosting on the similarities the off-the-shelf embeddings are best. With OLS on the similarities (not in the table), the fine-tuned text + image similarities carry less Q information than those of DoubleMLDeep and of the off-the-shelf model ($R^2$ 27.3 against 47.3 and 30.3; text only 42.0 against 46.2 and 31.2).

Inverse-rank elasticities with 90% CIs (`outputs/04_evaluation_comparison/ate_estimates.csv`, selected rows), $\chi^2$ test of heterogeneity along the clusters (`chi2_tests.csv`, PLR specification) and sensitivity analysis of the PLR with embeddings + controls (`sensitivity_robustness_values.csv`, $H_0: \theta = 0$; `sensitivity_calibrated.csv`; `sensitivity_main_shares.csv`); these files are in the output directory of each run, as listed in `code/04_evaluation_comparison.ipynb`:

| | DoubleMLDeep | EmbeddingGemma 2 | fine-tuned (txt+img, 256d) | fine-tuned (txt, 256d) | fine-tuned (txt+img, 768d) |
| --- | --- | --- | --- | --- | --- |
| Linear, lagged + controls + embeddings | -0.712 [-0.776, -0.649] | -0.731 [-0.795, -0.667] | -0.720 [-0.784, -0.657] | -0.705 [-0.769, -0.642] | -0.708 [-0.772, -0.645] |
| Linear, lagged + controls + similarities | -0.723 [-0.786, -0.659] | -0.735 [-0.799, -0.670] | -0.765 [-0.830, -0.700] | -0.730 [-0.794, -0.667] | -0.765 [-0.829, -0.700] |
| PLR, embeddings + controls | -0.697 [-0.777, -0.617] | -0.701 [-0.776, -0.625] | -0.667 [-0.743, -0.591] | -0.677 [-0.741, -0.614] | -0.675 [-0.750, -0.599] |
| PLR, similarities + controls | -0.691 [-0.757, -0.624] | -0.684 [-0.749, -0.620] | -0.758 [-0.833, -0.682] | -0.701 [-0.765, -0.636] | -0.742 [-0.813, -0.671] |
| PLR, PCA + controls | -0.677 [-0.749, -0.605] | -0.689 [-0.757, -0.622] | -0.732 [-0.805, -0.659] | -0.697 [-0.761, -0.632] | -0.715 [-0.783, -0.646] |
| $\chi^2$ p-value, heterogeneity along clusters (PLR) | 0.0009 | 0.0116 | 0.0020 | 0.0044 | 0.0003 |
| Robustness value RV (PLR, embeddings + controls) | 0.149 | 0.147 | 0.145 | 0.145 | 0.148 |
| Own calibration: residual share; upper 90% bound at 1x / 2x | 1.53%; -0.567 / -0.498 | 0.99%; -0.597 / -0.552 | 0.31%; -0.594 / -0.580 | 0.73%; -0.595 / -0.563 | 1.14%; -0.567 / -0.517 |
| Upper 90% bound at 1x / 2x the main calibration (1.53%, 3.06%) | -0.567 / -0.498 | -0.572 / -0.502 | -0.541 / -0.473 | -0.560 / -0.491 | -0.550 / -0.482 |

The average elasticities are close to those of the main analysis. With the fine-tuned text + image embeddings (256d and 768d), the specifications with cluster similarities or principal components give somewhat larger (more negative) elasticities than with the DoubleMLDeep embeddings (by 0.04 to 0.07); for the text-only run the differences are small (0.01 to 0.02). The PLR specification with embeddings + controls instead gives slightly smaller elasticities in all fine-tuned runs (-0.667 to -0.677 against -0.697). The heterogeneity along the clusters is significant in all runs (PLR, $p \le 0.012$). The observed-covariate benchmark of the sensitivity analysis is weaker with the fine-tuned embeddings (its alignment $|\tilde\rho|$ is 0.03 in the main specification, 0.07 text only and 0.12 at 768 dimensions, against 0.15 for DoubleMLDeep), so the own calibration is milder. At the residual shares of the main analysis, the upper bounds are slightly less negative than in the main analysis (by 0.007 to 0.026), mainly because the point estimate of the PLR specification with embeddings + controls is less negative; they stay below $-0.5$ at 1x (1.53%) and, as in the main analysis, lie above $-0.5$ at 2x (3.06%).

Clusters of the main specification on the estimation sample (the product-period rows of $I_2$ with an elasticity estimate: 3,708 products, 38,041 rows; `outputs/R2/cluster_summary_table.csv`, computed as in `main/code/02_cluster_summary.ipynb`). The cluster similarities are re-estimated for every embedding, so the clusters differ from those of the paper. Observations are assigned to the most similar cluster; the product counts printed in `02_cluster_centroid_products.ipynb` are larger because they pool $I_1$ and $I_2$. The median sales rank is the median across products of each product's sales rank in its first period in the sample (truncated to whole ranks, as in `main`). The last column is the mean over the cluster's observations of the observation-level CATEs of the PLR model with embeddings + controls (`SortEff_deep_dml` in `outputs/04_evaluation/sorted_effects.csv`), not a group-level estimate:

| Cluster | Label | # Products | Mean price ($) | Median sales rank | Mean inverse-rank elasticity (CATE) |
| --- | --- | --- | --- | --- | --- |
| 0 | Fingerboard Parts & Accessories | 552 | 15.15 | 101,650 | -0.507 |
| 1 | Licensed Character Track Playsets | 771 | 44.41 | 71,219 | -0.810 |
| 2 | Toddler Carrier Truck Sets | 891 | 27.00 | 64,852 | -0.635 |
| 3 | 1:64 Collectible Die-Cast | 824 | 14.68 | 89,867 | -0.723 |
| 4 | Alloy Pull-Back Model Cars | 670 | 22.97 | 90,474 | -0.622 |
