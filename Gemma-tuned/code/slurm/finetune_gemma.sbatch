#!/bin/bash
# LoRA fine-tuning of EmbeddingGemma 2 on a GPU node (alternative to 00_finetune_gemma_colab.ipynb).
#
# Usage (from Gemma-tuned/code):  sbatch slurm/finetune_gemma.sbatch txtimg
#                                 sbatch slurm/finetune_gemma.sbatch txt
# Before the first run (login node with internet access):
#   python -m venv $VENV && source $VENV/bin/activate && pip install -r requirements.txt
#   (torch==2.14.1 from PyPI is a CUDA 13 build that needs NVIDIA driver >= 580; with an older driver install
#   torch==2.14.1 and torchvision==0.29.1 first from the PyTorch index of a CUDA version the driver supports,
#   https://download.pytorch.org/whl/cuXXX, see https://pytorch.org/get-started/previous-versions/)
#   export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
#   python -c "from huggingface_hub import snapshot_download; snapshot_download('google/embeddinggemma-2', revision='914f7f89142e33e77833254d9c9b90c3cef7303b')"
# Adjust the #SBATCH lines (partition, GPU type) and the module loads to the cluster.
#SBATCH --job-name=gemma-tuned
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=gemma_tuned_%x_%j.log

set -euo pipefail
MODALITY=${1:-txtimg}
VENV=${VENV:-$HOME/venvs/gemma-tuned}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}  # compute nodes often have no internet access
export TOKENIZERS_PARALLELISM=false
RUN_ROOT=${RUN_ROOT:-$PWD/../runs}

# module load cuda  # site-specific
source "$VENV/bin/activate"
nvidia-smi
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'torch ' + torch.__version__ + ' cannot use the GPU (driver too old for its CUDA build?)')"
python finetune_gemma.py --modality "$MODALITY" --run-dir "$RUN_ROOT/$MODALITY" --num-workers 6
