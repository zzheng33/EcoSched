#!/usr/bin/env bash

set -euo pipefail

# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 ./simpleMultiGPU.sh [NUM_GPUS] [benchmark args...]
# Uses all visible GPUs automatically. MAX_ITERS controls how many times the binary is launched.

MAX_ITERS=1

NUM_GPUS="${NUM_GPUS:-4}"
if [[ $# -gt 0 && "$1" =~ ^[1-9][0-9]*$ ]]; then
    NUM_GPUS="$1"
    shift
fi

if [[ ! "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: NUM_GPUS must be a positive integer." >&2
    exit 1
fi

if [[ ! "$MAX_ITERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: MAX_ITERS must be a positive integer." >&2
    exit 1
fi

if ! command -v module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/modules.sh ]]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/modules.sh
    fi
fi

module use /soft/modulefiles
module load cuda/12.3.0
module load gcc/12.2.0
module load openmpi/4.1.1-gcc
module load public_mkl/2019

export CUDA_DIR=/soft/compilers/cuda/cuda-12.3.0
export PCM_NO_MSR=1
export PCM_KEEP_NMI_WATCHDOG=1
export LD_LIBRARY_PATH="${CUDA_DIR}/targets/x86_64-linux/lib:${CUDA_DIR}/lib64:${LD_LIBRARY_PATH:-}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
fi

IFS=',' read -r -a gpu_list <<< "$CUDA_VISIBLE_DEVICES"
if (( NUM_GPUS > ${#gpu_list[@]} )); then
    echo "Error: requested $NUM_GPUS GPUs but CUDA_VISIBLE_DEVICES only exposes ${#gpu_list[@]}." >&2
    exit 1
fi

selected_gpus=$(IFS=,; echo "${gpu_list[*]:0:NUM_GPUS}")
export CUDA_VISIBLE_DEVICES="$selected_gpus"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_BUILD_ROOT="${BENCHMARK_BUILD_ROOT:-$SCRIPT_DIR/../build-sm90}"
BENCHMARK_BIN="${BENCHMARK_BUILD_ROOT}/simpleMultiGPU/simpleMultiGPU"

if [[ ! -x "${BENCHMARK_BIN}" ]]; then
    echo "Error: benchmark binary not found or not executable: ${BENCHMARK_BIN}" >&2
    exit 1
fi

for ((iter = 1; iter <= MAX_ITERS; iter++)); do
    "${BENCHMARK_BIN}" "$@"
done
