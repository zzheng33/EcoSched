#!/bin/bash
export TOKENIZERS_PARALLELISM=false
set -euo pipefail

nproc_per_node="${1:-2}"

source "${HOME}/env/ml/bin/activate"
cd "${HOME}/benchmark/ECP/gpt2"

torchrun --nproc_per_node="${nproc_per_node}" training.py

deactivate
