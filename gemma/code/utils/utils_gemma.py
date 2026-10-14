import io
import json
import glob
import os
import time
import platform

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image


MODEL_ID = "google/embeddinggemma-2"
# Hugging Face commit of google/embeddinggemma-2 used for the reported results
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
# Symmetric task prefix recommended by the model card for clustering
PROMPT = "task: clustering | query: "
# Matryoshka dimensions supported by EmbeddingGemma 2
MRL_DIMS = (768, 512, 256, 128)


def load_product_texts(dataset_paths):
    """
    Collect the (time-invariant) product text for every ASIN from the processed
    datasets. Returns a DataFrame with columns ASIN and text, sorted by ASIN.
    """
    df = pd.concat(
        [pd.read_csv(path, usecols=["ASIN", "text"]) for path in dataset_paths],
        axis=0,
    )
    n_text_per_asin = df.groupby("ASIN")["text"].nunique()
    assert n_text_per_asin.max() == 1, "Expected exactly one text per ASIN"
    df = df.drop_duplicates("ASIN").sort_values("ASIN").reset_index(drop=True)
    return df


def load_product_image_bytes(image_dir, asins):
    """
    Read the raw (encoded) product images for the given ASINs.
    Returns a dict ASIN -> bytes. Images are decoded lazily per batch to keep
    memory usage low.
    """
    asins = set(asins)
    image_bytes = {}
    for path in sorted(glob.glob(os.path.join(image_dir, "*.parquet"))):
        table = pq.read_table(path).to_pandas()
        table = table[table["ASIN"].isin(asins)]
        for asin, image in zip(table["ASIN"], table["image"]):
            image_bytes[asin] = image["bytes"]
    missing = asins - set(image_bytes)
    assert len(missing) == 0, f"Missing images for {len(missing)} ASINs"
    return image_bytes


def load_model(dtype="bfloat16", modality="txtimg", revision=MODEL_REVISION):
    """
    Load EmbeddingGemma 2 via sentence-transformers. Unused modality encoders
    are not loaded (see model card, "Selective Encoder Loading").
    """
    import torch
    from sentence_transformers import SentenceTransformer

    config_kwargs = {"audio_config": None}
    if modality == "txt":
        config_kwargs["vision_config"] = None

    torch_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    model = SentenceTransformer(
        MODEL_ID,
        revision=revision,
        device="cpu" if not torch.cuda.is_available() else "cuda",
        model_kwargs={"torch_dtype": torch_dtype},
        config_kwargs=config_kwargs,
    )
    return model


def build_inputs(texts, images=None):
    """
    Text-only inputs get the clustering prefix. For text+image inputs the image
    is interleaved after the text via the <|image|> placeholder; the prefix
    applies to the text only (see model card, "Multimodal Input").
    """
    if images is None:
        return [PROMPT + text for text in texts]
    return [
        {"text": PROMPT + text + " <|image|>", "image": [image]}
        for text, image in zip(texts, images)
    ]


def compute_gemma_embeddings(
    df_products,
    modality="txtimg",
    image_bytes=None,
    dtype="bfloat16",
    batch_size=8,
    chunk_size=256,
    cache_dir=None,
    num_threads=None,
):
    """
    Compute L2-normalized 768-dimensional EmbeddingGemma 2 embeddings for each
    product. Work is split into chunks that are cached as .npy files in
    `cache_dir`, so that an interrupted run can be resumed.

    Returns a DataFrame indexed by ASIN with float32 columns emb_0, ..., emb_767.
    """
    import torch

    if num_threads is not None:
        torch.set_num_threads(num_threads)
    if modality == "txtimg":
        assert image_bytes is not None, "image_bytes required for modality='txtimg'"
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    asins = df_products["ASIN"].tolist()
    texts = df_products["text"].tolist()
    n_chunks = int(np.ceil(len(asins) / chunk_size))

    model = None
    chunks = []
    t_start = time.time()
    for i_chunk in range(n_chunks):
        lo, hi = i_chunk * chunk_size, min((i_chunk + 1) * chunk_size, len(asins))
        cache_file = (
            os.path.join(cache_dir, f"{modality}_{dtype}_chunk{i_chunk:04d}.npz")
            if cache_dir is not None
            else None
        )
        if cache_file is not None and os.path.exists(cache_file):
            cached = np.load(cache_file, allow_pickle=False)
            assert list(cached["asins"]) == asins[lo:hi], "Cache does not match input"
            chunks.append(cached["embeddings"])
            continue

        if model is None:
            model = load_model(dtype=dtype, modality=modality)

        images = None
        if modality == "txtimg":
            images = [
                Image.open(io.BytesIO(image_bytes[asin])).convert("RGB")
                for asin in asins[lo:hi]
            ]
        inputs = build_inputs(texts[lo:hi], images)
        embeddings = model.encode(
            inputs,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype(np.float32)
        # normalize in float32 (bfloat16 outputs are only approximately unit length)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        assert np.isfinite(embeddings).all(), "Non-finite embeddings encountered"

        if cache_file is not None:
            np.savez(cache_file, asins=np.array(asins[lo:hi]), embeddings=embeddings)
        chunks.append(embeddings)

        elapsed = time.time() - t_start
        print(
            f"[{modality}] chunk {i_chunk + 1}/{n_chunks} done "
            f"({hi}/{len(asins)} products, {elapsed / 60:.1f} min elapsed)",
            flush=True,
        )

    embeddings = np.concatenate(chunks, axis=0)
    df_embeddings = pd.DataFrame(
        embeddings,
        index=pd.Index(asins, name="ASIN"),
        columns=[f"emb_{i}" for i in range(embeddings.shape[1])],
    )
    return df_embeddings


def embedding_metadata(modality, dtype, n_products):
    import torch
    import transformers
    import sentence_transformers

    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "modality": modality,
        "prompt": PROMPT,
        "input_format": (
            "PROMPT + text + ' <|image|>' (image interleaved after text)"
            if modality == "txtimg"
            else "PROMPT + text"
        ),
        "vision_soft_tokens_per_image": 280 if modality == "txtimg" else None,
        "dtype": dtype,
        "pooling": "mean (model default), L2-normalized in float32",
        "embedding_dim": 768,
        "n_products": int(n_products),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "python": platform.python_version(),
        "machine": platform.processor() or platform.machine(),
    }


def save_embeddings(df_embeddings, path, metadata=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_embeddings.reset_index().to_parquet(path, index=False)
    if metadata is not None:
        with open(os.path.splitext(path)[0] + ".json", "w") as f:
            json.dump(metadata, f, indent=2)


def load_embeddings(path, emb_dim=768):
    """
    Load product-level embeddings and apply Matryoshka truncation to the
    leading `emb_dim` dimensions followed by L2 re-normalization (model card,
    "Matryoshka Dimension Truncation").
    """
    assert emb_dim in MRL_DIMS, f"emb_dim must be one of {MRL_DIMS}"
    df = pd.read_parquet(path).set_index("ASIN")
    values = df.values[:, :emb_dim].astype(np.float64)
    values = values / np.linalg.norm(values, axis=1, keepdims=True)
    return pd.DataFrame(
        values, index=df.index, columns=[f"emb_{i}" for i in range(emb_dim)]
    )
