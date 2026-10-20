"""
LoRA fine-tuning of EmbeddingGemma 2 with the DoubleMLDeep loss.

Mirrors the training of the (time-independent) DoubleMLDeep model whose embeddings are used in main/:
- products of the HF "train" split (I_1) are used for the gradient updates,
- the loss is the product of the root-mean-square errors of the two targets, computed over the
  (product, period) rows of a mini-batch: sqrt(MSE_Q + 1e-8) * sqrt(MSE_P + 1e-8),
- the checkpoint (epoch) is selected and the learning-rate schedule is driven by the same loss over all
  rows of the HF "validation" split (I_2), as in the original run (val_combined_loss),
- targets are SALES_RANK (Q_t, -log sales rank) and BUYBOX_PRICE (P_bb_t, log price), without scaling,
- the heads are LayerNorm followed by two separate Linear(d, 128) -> Linear(128, 1) maps (no activation),
  as in the DoubleMLDeep FusionHead with hidden_dim=[128].

The product inputs (text and image) are time-invariant, so the model is evaluated once per product and the
row-level loss is computed exactly from per-product sufficient statistics (number of rows, mean and
within-product sum of squares of each target).
"""
import io
import json
import math
import os
import platform
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from utils.utils_gemma import (
    MODEL_ID,
    MODEL_REVISION,
    PROMPT,
    build_inputs,
    load_product_image_bytes,
)

TARGETS = ("Q_t", "P_bb_t")  # outcome (ml_l, SALES_RANK) and treatment (ml_m, BUYBOX_PRICE)
# LoRA target modules (regular expressions, matched with re.fullmatch by PEFT). The vision projections are
# wrapped in Gemma4ClippableLinear, hence the ".linear" suffix.
LORA_TEXT_MODULES = r"language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
LORA_VISION_MODULES = (
    r"vision_tower\.encoder\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.linear"
)


@dataclass
class FinetuneConfig:
    modality: str = "txtimg"  # "txtimg" (text + image) or "txt" (text only)
    emb_dim: int = 256  # Matryoshka dimension used by the head and downstream (01_1 / 01_2)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_vision: bool = True  # also adapt the vision encoder (txtimg only)
    head_hidden: int = 128
    batch_size: int = 16  # products per mini-batch (about 11 rows per product)
    eval_batch_size: int = 32
    lr_lora: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    max_epochs: int = 30
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    lr_factor: float = 0.1  # ReduceLROnPlateau on the I_2 loss (DoubleMLDeep default: factor 0.1, patience 0)
    lr_patience: int = 0
    grad_clip: float = 1.0
    gradient_checkpointing: bool = True
    dtype: str = "auto"  # "auto": bfloat16 on CPU and on GPUs with compute capability >= 8, else float32
    seed: int = 42
    num_workers: int = 2
    # limits for smoke tests (None = all)
    max_train_products: Optional[int] = None
    max_val_products: Optional[int] = None
    max_steps_per_epoch: Optional[int] = None
    max_export_products: Optional[int] = None

    def to_json(self):
        return asdict(self)


# ---------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------
def load_finetune_data(main_data_dir):
    """
    Rows (ASIN, date) of I_1 ("train") and I_2 ("val") with the targets, as in the datasets of main/
    (window 28, every 4th period, 2023-01-30 to 2024-01-29), the time-invariant product texts and the
    per-product sufficient statistics of the targets.
    """
    frames = []
    for split in ["train", "val"]:
        df = pd.read_csv(
            os.path.join(main_data_dir, f"dataset_txt_only_False_embeddings_True_{split}.zip"),
            usecols=["ASIN", "date", "text", *TARGETS],
        )
        df["split"] = split
        frames.append(df)
    rows = pd.concat(frames, ignore_index=True)
    assert rows[list(TARGETS)].notna().all().all(), "missing targets"
    assert rows.groupby("ASIN")["split"].nunique().max() == 1, "the split must be product-level"
    assert rows.groupby("ASIN")["text"].nunique().max() == 1, "expected one text per product"

    products = (
        rows.drop_duplicates("ASIN")[["ASIN", "split", "text"]].sort_values("ASIN").reset_index(drop=True)
    )
    g = rows.groupby("ASIN")
    stats = pd.DataFrame({"n": g.size()})
    for target in TARGETS:
        stats[f"{target}_mean"] = g[target].mean()
        stats[f"{target}_ss"] = g[target].apply(lambda x: float(((x - x.mean()) ** 2).sum()))
    stats = stats.loc[products["ASIN"]]
    rows = rows.drop(columns="text").sort_values(["ASIN", "date"]).reset_index(drop=True)
    return products, rows, stats


def load_images(main_data_dir, asins):
    return load_product_image_bytes(
        os.path.join(main_data_dir, "amzn_toys_monthly_avg_long_ffill_28_7_images"), asins
    )


class ProductCollator:
    """Collate a list of ASINs into model features (tokenization and image processing)."""

    def __init__(self, st_model, texts, image_bytes, modality):
        self.st_model = st_model
        self.texts = texts  # dict ASIN -> text
        self.image_bytes = image_bytes  # dict ASIN -> encoded image (txtimg only)
        self.modality = modality

    def __call__(self, asins):
        texts = [self.texts[a] for a in asins]
        images = None
        if self.modality == "txtimg":
            images = [Image.open(io.BytesIO(self.image_bytes[a])).convert("RGB") for a in asins]
        features = self.st_model.preprocess(build_inputs(texts, images))
        return list(asins), features


# ---------------------------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------------------------
def resolve_dtype(name):
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            # torch.cuda.is_bf16_supported() also returns True for emulated bfloat16 (e.g. T4); the model must
            # not run in float16 (model card), so fall back to float32 without native bfloat16
            major, _ = torch.cuda.get_device_capability()
            return torch.bfloat16 if major >= 8 else torch.float32
        return torch.bfloat16
    return {"bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def build_model(cfg, device, dtype, target_means=None):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from sentence_transformers import SentenceTransformer

    config_kwargs = {"audio_config": None}
    if cfg.modality == "txt":
        config_kwargs["vision_config"] = None
    st = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        device=str(device),
        model_kwargs={"torch_dtype": dtype},
        config_kwargs=config_kwargs,
    )
    target_modules = LORA_TEXT_MODULES
    if cfg.modality == "txtimg" and cfg.lora_vision:
        target_modules = f"({LORA_TEXT_MODULES})|({LORA_VISION_MODULES})"
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    st[0].model = get_peft_model(st[0].model, lora_config)
    if cfg.gradient_checkpointing:
        st[0].model.base_model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    class GemmaTuned(nn.Module):
        def __init__(self):
            super().__init__()
            self.st = st
            self.emb_dim = cfg.emb_dim
            # shared LayerNorm of the embedding, then one head per target
            self.norm = nn.LayerNorm(cfg.emb_dim)
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(cfg.emb_dim, cfg.head_hidden), nn.Linear(cfg.head_hidden, 1))
                    for _ in TARGETS
                ]
            )
            for head, mean in zip(self.heads, target_means or [0.0] * len(TARGETS)):
                for layer in head:
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
                # start the predictions at the I_1 mean of the target (targets are not scaled)
                nn.init.constant_(head[-1].bias, float(mean))

        def embed(self, features):
            # 768-d mean-pooled, L2-normalized EmbeddingGemma 2 output; Matryoshka truncation to emb_dim,
            # re-normalized (the representation used downstream)
            out = self.st(features)["sentence_embedding"].float()
            z = F.normalize(out[:, : self.emb_dim], dim=-1)
            return out, z

        def predict(self, z):
            h = self.norm(z)
            return torch.cat([head(h) for head in self.heads], dim=1)

        def forward(self, features):
            out, z = self.embed(features)
            return self.predict(z), out

    model = GemmaTuned().to(device)
    return model


def trainable_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def load_trainable_state(model, state):
    params = dict(model.named_parameters())
    assert set(state) == {n for n, p in params.items() if p.requires_grad}, "checkpoint does not match the model"
    import torch

    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value.to(params[name].device, params[name].dtype))


def move_features(features, device):
    import torch

    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in features.items()}


# ---------------------------------------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------------------------------------
def rmse_product(pred, n, mean, ss, eps=1e-8):
    """
    Product of the row-level RMSEs of the two targets for a set of products with time-invariant predictions:
    MSE_k = sum_i [ n_i (mean_ik - pred_ik)^2 + ss_ik ] / sum_i n_i.
    pred, mean, ss: [B, 2]; n: [B]. Returns the loss and the two RMSEs.
    """
    import torch

    sse = (n[:, None] * (mean - pred) ** 2 + ss).sum(dim=0)
    rmse = torch.sqrt(sse / n.sum() + eps)
    return rmse[0] * rmse[1], rmse


def stats_tensors(stats, asins, device):
    import torch

    s = stats.loc[asins]
    n = torch.tensor(s["n"].values, dtype=torch.float32, device=device)
    mean = torch.tensor(s[[f"{t}_mean" for t in TARGETS]].values, dtype=torch.float32, device=device)
    ss = torch.tensor(s[[f"{t}_ss" for t in TARGETS]].values, dtype=torch.float32, device=device)
    return n, mean, ss


# ---------------------------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------------------------
def predict_products(model, asins, collator, cfg, device, return_embeddings=False, log_every=None):
    """Predictions (and 768-d embeddings) for the given products, in eval mode, batches sorted by text length."""
    import torch
    from torch.utils.data import DataLoader

    order = sorted(asins, key=lambda a: len(collator.texts[a]))
    loader = DataLoader(
        order,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        persistent_workers=False,
    )
    model.eval()
    preds, embs, names = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, (batch_asins, features) in enumerate(loader):
            pred, out = model(move_features(features, device))
            preds.append(pred.float().cpu().numpy())
            if return_embeddings:
                out = out.float().cpu().numpy()
                embs.append(out / np.linalg.norm(out, axis=1, keepdims=True))  # re-normalize in float32
            names += batch_asins
            if log_every and (i + 1) % log_every == 0:
                print(f"  predicted {len(names)}/{len(order)} products ({time.time() - t0:.0f} s)", flush=True)
    pred = pd.DataFrame(np.concatenate(preds), index=pd.Index(names, name="ASIN"), columns=["pred_ml_l", "pred_ml_m"])
    emb = None
    if return_embeddings:
        emb = pd.DataFrame(
            np.concatenate(embs), index=pred.index, columns=[f"emb_{i}" for i in range(embs[0].shape[1])]
        )
        emb = emb.loc[asins]
    return pred.loc[asins], emb


def evaluate_loss(pred, stats):
    """Whole-split loss (as val_combined_loss), RMSEs and R^2 of the two targets from product predictions."""
    s = stats.loc[pred.index]
    out = {}
    rmse = []
    for target, col in zip(TARGETS, ["pred_ml_l", "pred_ml_m"]):
        sse = (s["n"] * (s[f"{target}_mean"] - pred[col]) ** 2 + s[f"{target}_ss"]).sum()
        grand_mean = (s["n"] * s[f"{target}_mean"]).sum() / s["n"].sum()
        sst = (s["n"] * (s[f"{target}_mean"] - grand_mean) ** 2 + s[f"{target}_ss"]).sum()
        r = math.sqrt(sse / s["n"].sum() + 1e-8)
        rmse.append(r)
        out[f"rmse_{target}"] = r
        out[f"r2_{target}"] = 1.0 - sse / sst
    out["loss"] = rmse[0] * rmse[1]
    return out


# ---------------------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------------------
def set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_info(device, dtype):
    import peft
    import sentence_transformers
    import torch
    import transformers

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "peft": peft.__version__,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def run_finetuning(cfg, main_data_dir, out_dir, resume=True, log=print):
    """
    Train (or resume) and return the trained model (best checkpoint loaded), the data and the history.
    Writes to out_dir: config.json, history.json, checkpoint_last.pt, checkpoint_best.pt.
    """
    import torch
    from torch.utils.data import DataLoader

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(cfg.dtype)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    set_seed(cfg.seed)

    products, rows, stats = load_finetune_data(main_data_dir)
    train_asins = products.loc[products["split"] == "train", "ASIN"].tolist()
    val_asins = products.loc[products["split"] == "val", "ASIN"].tolist()
    if cfg.max_train_products:
        train_asins = train_asins[: cfg.max_train_products]
    if cfg.max_val_products:
        val_asins = val_asins[: cfg.max_val_products]
    log(f"I_1: {len(train_asins)} products, I_2: {len(val_asins)} products")

    image_bytes = None
    if cfg.modality == "txtimg":
        image_bytes = load_images(main_data_dir, train_asins + val_asins)
    target_means = [
        float((stats.loc[train_asins, "n"] * stats.loc[train_asins, f"{t}_mean"]).sum() / stats.loc[train_asins, "n"].sum())
        for t in TARGETS
    ]
    model = build_model(cfg, device, dtype, target_means=target_means)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"trainable parameters: {n_trainable:,} ({dtype}, {device})")
    collator = ProductCollator(model.st, dict(zip(products["ASIN"], products["text"])), image_bytes, cfg.modality)

    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    optimizer = torch.optim.AdamW(
        [{"params": lora_params, "lr": cfg.lr_lora}, {"params": head_params, "lr": cfg.lr_head}],
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience
    )

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"config": cfg.to_json(), "environment": environment_info(device, dtype),
                   "n_trainable": n_trainable, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                   "prompt": PROMPT}, f, indent=2)

    state = {"epoch": 0, "best_loss": float("inf"), "best_epoch": None, "es_best": float("inf"), "es_bad": 0,
             "history": [], "stopped": False}
    last_path = os.path.join(out_dir, "checkpoint_last.pt")
    best_path = os.path.join(out_dir, "checkpoint_best.pt")
    if resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        load_trainable_state(model, ckpt["trainable"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        state = ckpt["state"]
        log(f"resumed after epoch {state['epoch']} (best I_2 loss {state['best_loss']:.4f} at epoch {state['best_epoch']})")

    while state["epoch"] < cfg.max_epochs and not state["stopped"]:
        epoch = state["epoch"] + 1
        model.train()
        generator = torch.Generator().manual_seed(cfg.seed + epoch)  # same order when resuming
        loader = DataLoader(
            train_asins,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=len(train_asins) > cfg.batch_size,
            generator=generator,
            collate_fn=collator,
            num_workers=cfg.num_workers,
        )
        t0, sum_loss, n_steps, n_prod = time.time(), 0.0, 0, 0
        for step, (asins, features) in enumerate(loader):
            if cfg.max_steps_per_epoch and step >= cfg.max_steps_per_epoch:
                break
            pred, _ = model(move_features(features, device))
            n, mean, ss = stats_tensors(stats, asins, device)
            loss, _ = rmse_product(pred.float(), n, mean, ss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
            optimizer.step()
            sum_loss += loss.item()
            n_steps += 1
            n_prod += len(asins)
            if n_steps % 20 == 0 or n_steps == 1:
                rate = n_prod / (time.time() - t0)
                mem = f", GPU mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB" if device.type == "cuda" else ""
                log(f"epoch {epoch} step {n_steps}/{len(loader)}: batch loss {loss.item():.4f}, "
                    f"{rate:.1f} products/s{mem}")
        train_time = time.time() - t0

        t1 = time.time()
        pred_val, _ = predict_products(model, val_asins, collator, cfg, device)
        val = evaluate_loss(pred_val, stats)
        val_time = time.time() - t1
        scheduler.step(val["loss"])
        record = {
            "epoch": epoch,
            "train_batch_loss_mean": sum_loss / max(n_steps, 1),
            **{f"val_{k}": v for k, v in val.items()},
            "lr_lora": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
            "train_seconds": train_time,
            "val_seconds": val_time,
        }
        state["history"].append(record)
        log(f"epoch {epoch}: train batch loss {record['train_batch_loss_mean']:.4f} | I_2 loss {val['loss']:.4f} "
            f"(RMSE_Q {val['rmse_Q_t']:.4f}, RMSE_P {val['rmse_P_bb_t']:.4f}, R2_Q {val['r2_Q_t']:.3f}, "
            f"R2_P {val['r2_P_bb_t']:.3f}) | {train_time / 60:.1f} + {val_time / 60:.1f} min")

        improved = val["loss"] < state["best_loss"]
        if improved:  # checkpoint selection: minimum I_2 loss
            state["best_loss"], state["best_epoch"] = val["loss"], epoch
            torch.save({"trainable": trainable_state(model), "epoch": epoch, "val": val}, best_path + ".tmp")
            os.replace(best_path + ".tmp", best_path)
        if val["loss"] < state["es_best"] - cfg.early_stopping_min_delta:  # early stopping
            state["es_best"], state["es_bad"] = val["loss"], 0
        else:
            state["es_bad"] += 1
            if state["es_bad"] >= cfg.early_stopping_patience:
                state["stopped"] = True
                log(f"early stopping after epoch {epoch}")
        state["epoch"] = epoch
        torch.save({"trainable": trainable_state(model), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "state": state}, last_path + ".tmp")
        os.replace(last_path + ".tmp", last_path)
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump({"best_epoch": state["best_epoch"], "best_val_loss": state["best_loss"],
                       "history": state["history"]}, f, indent=2)

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    load_trainable_state(model, best["trainable"])
    log(f"loaded best checkpoint: epoch {best['epoch']}, I_2 loss {best['val']['loss']:.4f}")
    data = {"products": products, "rows": rows, "stats": stats, "collator": collator, "device": device,
            "dtype": dtype, "train_asins": train_asins, "val_asins": val_asins}
    return model, data, state


def export_results(model, data, cfg, out_dir, export_dir, log=print):
    """
    Write the fine-tuned outputs to export_dir:
    - embeddinggemma2_tuned_<modality>.parquet / .json: 768-d L2-normalized embeddings of all products (I_1 and
      I_2; Matryoshka truncation to emb_dim and re-normalization as in gemma/ reproduces the head input)
    - predictions_<modality>.parquet: pred_ml_l, pred_ml_m for every (ASIN, date) row of I_1 and I_2
    - adapter_<modality>/ (PEFT LoRA adapter), head_<modality>.pt, history and config of the run
    """
    import torch

    os.makedirs(export_dir, exist_ok=True)
    products, rows, stats = data["products"], data["rows"], data["stats"]
    asins = products["ASIN"].tolist()
    if cfg.max_export_products:
        asins = data["train_asins"][: cfg.max_export_products // 2] + data["val_asins"][: cfg.max_export_products // 2]
    t0 = time.time()
    pred, emb = predict_products(model, asins, data["collator"], cfg, data["device"], return_embeddings=True,
                                 log_every=20)
    log(f"exported {len(asins)} products in {(time.time() - t0) / 60:.1f} min")
    assert np.isfinite(emb.values).all() and np.isfinite(pred.values).all()

    tag = cfg.modality
    emb_path = os.path.join(export_dir, f"embeddinggemma2_tuned_{tag}.parquet")
    emb.reset_index().to_parquet(emb_path, index=False)
    history = json.load(open(os.path.join(out_dir, "history.json")))
    split = products.set_index("ASIN")["split"]
    metrics = {s: evaluate_loss(pred[split.loc[pred.index] == s], stats) for s in ["train", "val"]
               if (split.loc[pred.index] == s).any()}
    meta = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "modality": tag,
        "prompt": PROMPT,
        "fine_tuning": "LoRA, DoubleMLDeep loss RMSE(Q)*RMSE(P) on I_1, epoch selected on I_2",
        "head_input": f"Matryoshka truncation to {cfg.emb_dim} dimensions, L2-normalized",
        "embedding_dim": int(emb.shape[1]),
        "n_products": int(len(emb)),
        "best_epoch": history["best_epoch"],
        "best_val_loss": history["best_val_loss"],
        "loss_by_split": metrics,
        "config": cfg.to_json(),
        "environment": environment_info(data["device"], data["dtype"]),
    }
    with open(os.path.splitext(emb_path)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)

    pred_rows = rows[rows["ASIN"].isin(pred.index)].join(pred, on="ASIN")
    pred_rows.to_parquet(os.path.join(export_dir, f"predictions_{tag}.parquet"), index=False)

    model.st[0].model.save_pretrained(os.path.join(export_dir, f"adapter_{tag}"))
    head_state = {k: v for k, v in trainable_state(model).items() if "lora_" not in k}
    torch.save(head_state, os.path.join(export_dir, f"head_{tag}.pt"))
    for name in ["history.json", "config.json"]:
        shutil.copy(os.path.join(out_dir, name), os.path.join(export_dir, f"{os.path.splitext(name)[0]}_{tag}.json"))
    log(f"wrote {export_dir}: loss by split {json.dumps({k: round(v['loss'], 4) for k, v in metrics.items()})}")
    return meta
