#!/usr/bin/env bash

set -euo pipefail

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

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_BUILD_ROOT="${BENCHMARK_BUILD_ROOT:-$SCRIPT_DIR/../build-sm90}"
BENCHMARK_BIN="${BENCHMARK_BUILD_ROOT}/simpleCUFFT_MGPU/simpleCUFFT_MGPU"

if [[ ! -x "${BENCHMARK_BIN}" ]]; then
    echo "Error: benchmark binary not found or not executable: ${BENCHMARK_BIN}" >&2
    exit 1
fi

exec "${BENCHMARK_BIN}" "$@"
