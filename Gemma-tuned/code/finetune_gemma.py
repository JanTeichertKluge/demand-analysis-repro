"""
LoRA fine-tuning of EmbeddingGemma 2 with the DoubleMLDeep loss (see utils/utils_finetune.py).

Run from Gemma-tuned/code (GPU recommended: about 5.4 h for both modalities on an A100, 4.4 h text + image and 1.0 h
text only; considerably longer on an L4):

    python finetune_gemma.py --modality txtimg --run-dir RUN_DIR
    python finetune_gemma.py --modality txt --run-dir RUN_DIR

RUN_DIR holds the checkpoints (resumed automatically if the run is interrupted). The final outputs are written to
--export-dir (default ../data/finetuned/<modality>). Use --smoke for a quick end-to-end test on a few products.
"""
import argparse
import dataclasses
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.utils_finetune import FinetuneConfig, export_results, run_finetuning  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modality", choices=["txtimg", "txt"], required=True)
    parser.add_argument("--run-dir", required=True, help="checkpoints and logs (use persistent storage)")
    parser.add_argument("--export-dir", default=None, help="default: ../data/finetuned/<modality>")
    parser.add_argument("--main-data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../main/data"))
    parser.add_argument("--no-resume", action="store_true", help="train from scratch (also if already exported)")
    parser.add_argument("--allow-cpu", action="store_true", help="allow a full run without a GPU")
    parser.add_argument("--smoke", action="store_true", help="tiny run (8+8 products, 2 epochs, 2 steps) to test the setup")
    for f in dataclasses.fields(FinetuneConfig):
        if f.name == "modality":
            continue
        kind = f.type if isinstance(f.type, type) else str
        if f.type in (bool, "bool"):
            parser.add_argument(f"--{f.name.replace('_', '-')}", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        elif "int" in str(f.type):
            parser.add_argument(f"--{f.name.replace('_', '-')}", type=int, default=None)
        elif "float" in str(f.type):
            parser.add_argument(f"--{f.name.replace('_', '-')}", type=float, default=None)
        else:
            parser.add_argument(f"--{f.name.replace('_', '-')}", type=kind, default=None)
    args = parser.parse_args()

    cfg = FinetuneConfig(modality=args.modality)
    if args.smoke:
        cfg = dataclasses.replace(cfg, max_train_products=8, max_val_products=8, max_steps_per_epoch=2,
                                  max_epochs=2, batch_size=4, eval_batch_size=4, max_export_products=8)
    for f in dataclasses.fields(FinetuneConfig):
        value = getattr(args, f.name, None)
        if f.name != "modality" and value is not None:
            cfg = dataclasses.replace(cfg, **{f.name: value})

    export_dir = args.export_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/finetuned", args.modality)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"config: {json.dumps(cfg.to_json())}")
    export_dir = os.path.abspath(export_dir)
    done_path = os.path.join(args.run_dir, "export_done.json")
    if not args.no_resume and os.path.exists(done_path):
        done = json.load(open(done_path))
        meta_path = os.path.join(export_dir, f"embeddinggemma2_tuned_{cfg.modality}.json")
        if done["config"] == cfg.to_json() and done["export_dir"] == export_dir and os.path.exists(meta_path):
            log(f"already trained and exported to {export_dir} (best epoch {done['best_epoch']}); skipping")
            return
    model, data, state = run_finetuning(cfg, args.main_data_dir, args.run_dir, resume=not args.no_resume,
                                        allow_cpu=args.allow_cpu, log=log)
    meta = export_results(model, data, cfg, args.run_dir, export_dir, log=log)
    if state["stopped"] or state["epoch"] >= cfg.max_epochs:
        with open(done_path, "w") as f:
            json.dump({"config": cfg.to_json(), "export_dir": export_dir, "best_epoch": meta["best_epoch"]}, f)
    log("done")


if __name__ == "__main__":
    main()
