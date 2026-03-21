#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

set -euo pipefail

nproc_per_node="${1:-2}"

source "${HOME}/env/ml/bin/activate"
cd "${HOME}/benchmark/ECP/gpt2"

rm -rf /tmp/torchelastic_* 2>/dev/null || true


torchrun --nproc_per_node="${nproc_per_node}" training.py

deactivate
