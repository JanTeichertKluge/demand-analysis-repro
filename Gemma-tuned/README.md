# Fine-Tuned EmbeddingGemma 2 Variant of the Main Analysis

This folder repeats the EmbeddingGemma 2 variant (`gemma/`) with an EmbeddingGemma 2 model that is first fine-tuned on our data with LoRA, in the same way as the DoubleMLDeep model whose embeddings are used in `main/`.

## Fine-tuning
- Model `google/embeddinggemma-2`, revision `914f7f8`, prompt `task: clustering | query: `, as in `gemma/`.
- Sample split of the DoubleMLDeep model: the products of the HF `train` split ($I_1$, 3,774 products, 41,945 product-period rows) are used for the gradient updates; the epoch is selected and the learning rate reduced on the loss over all rows of the `validation` split ($I_2$, 3,775 products, 41,832 rows).
- Loss of the DoubleMLDeep model: $\mathrm{RMSE}(Q)\cdot\mathrm{RMSE}(P)$ over the product-period rows of a mini-batch, with $Q$ = `SALES_RANK` ($-\log$ sales rank) and $P$ = `BUYBOX_PRICE` (log price). Inputs are time-invariant, so each product is embedded once per step and the row-level loss is computed exactly from per-product statistics.
- Heads as in the DoubleMLDeep model: LayerNorm, then $\mathrm{Linear}(256,128)\to\mathrm{Linear}(128,1)$ per target, on the Matryoshka truncation of the embedding to 256 dimensions (the representation used downstream). The last layer starts at zero and its bias at the $I_1$ target means, so training starts from the constant predictor.
- LoRA ($r=16$, $\alpha=32$, dropout 0.05) on the attention and MLP projections of the language model and, for text + image, of the vision encoder. AdamW (learning rate $10^{-4}$ for LoRA, $10^{-3}$ for the heads), ReduceLROnPlateau (factor 0.1, patience 1) on the $I_2$ loss, early stopping after 4 epochs without an improvement of at least $10^{-4}$, at most 20 epochs. The text + image run reached the 20-epoch limit while the $I_2$ loss was still decreasing slightly (0.2971 to 0.2966 in epoch 20, selected epoch 20); the text-only run stopped early after epoch 19 (selected epoch 19).
- Deliberate deviations from the DoubleMLDeep training: AdamW instead of SGD with momentum (standard for LoRA), and therefore a learning-rate patience of 1 instead of 0 (with AdamW the epoch-level $I_2$ loss fluctuates enough that patience 0 cuts the learning rate on noise); the zero-initialized last layer described above.
- Two runs: `txtimg` (text + image) and `txt` (text only, used where `main` uses text-only embeddings).

Reference values of the $I_2$ loss: constant predictor 0.818; DoubleMLDeep 0.332 (text + image) and 0.340 (text only); off-the-shelf EmbeddingGemma 2 at 256 dimensions with a ridge head 0.479 (text + image) and 0.500 (text only).

## Running the fine-tuning (GPU)
- Colab: `code/00_finetune_gemma_colab.ipynb` (A100 or L4 runtime, Colab secret `GITHUB_TOKEN`, *Run all*). Checkpoints are kept on Google Drive and a run resumes after a disconnect. The results are pushed to the branch `gemma-tuned-results`.
- Slurm: `code/slurm/finetune_gemma.sbatch` (see its header).
- Command line, from `code/`: `python finetune_gemma.py --modality txtimg --run-dir RUN_DIR` (and `--modality txt`). `--smoke` runs a short end-to-end test.

The code is in `code/finetune_gemma.py` and `code/utils/utils_finetune.py`. The outputs are written to `data/finetuned/<modality>/`: 768-dimensional L2-normalized embeddings of all products (`embeddinggemma2_tuned_<modality>.parquet`, with metadata `.json`), the predictions of the two heads for every product-period row, the LoRA adapter, the heads, and the training history.

## Status
The fine-tuning code is tested end to end on the CPU with a few products. The pipeline notebooks (01 to 04, as in `gemma/`) are run with the tuned embeddings once the GPU results are available.
