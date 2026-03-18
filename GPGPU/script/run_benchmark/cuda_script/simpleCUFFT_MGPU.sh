#!/usr/bin/env bash

set -euo pipefail

# Usage: CUDA_VISIBLE_DEVICES=0,1 ./simpleCUFFT_MGPU.sh [NUM_GPUS] [benchmark args...]
# Example: CUDA_VISIBLE_DEVICES=0,1,2,3 ./simpleCUFFT_MGPU.sh 4

NUM_GPUS="${1:-4}"
if [[ $# -gt 0 ]]; then
    shift
fi

if [[ ! "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: NUM_GPUS must be a positive integer." >&2
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
    echo "Error: CUDA_VISIBLE_DEVICES must be set externally before running this script." >&2
    exit 1
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
BENCHMARK_BIN="${BENCHMARK_BUILD_ROOT}/simpleCUFFT_MGPU/simpleCUFFT_MGPU"

if [[ ! -x "${BENCHMARK_BIN}" ]]; then
    echo "Error: benchmark binary not found or not executable: ${BENCHMARK_BIN}" >&2
    exit 1
fi

exec "${BENCHMARK_BIN}" "$@"
