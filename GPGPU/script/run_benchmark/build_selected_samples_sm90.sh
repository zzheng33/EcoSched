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
module load cmake
module load gcc/12.2.0
module load openmpi/4.1.1-gcc
module load public_mkl/2019
export CUDA_DIR=/soft/compilers/cuda/cuda-12.3.0
export CC="${CC:-$(command -v gcc)}"
export CXX="${CXX:-$(command -v g++)}"
export PCM_NO_MSR=1
export PCM_KEEP_NMI_WATCHDOG=1

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_SAMPLES_ROOT="/home/ac.zzheng/benchmark/cuda/Samples"
BUILD_ROOT="${BUILD_ROOT:-$SCRIPT_DIR/build-sm90}"
JOBS="${JOBS:-$(nproc)}"

SAMPLES=(
    "${BENCHMARK_SAMPLES_ROOT}/4_CUDA_Libraries/simpleCUFFT_2d_MGPU"
    "${BENCHMARK_SAMPLES_ROOT}/4_CUDA_Libraries/simpleCUFFT_MGPU"
    "${BENCHMARK_SAMPLES_ROOT}/4_CUDA_Libraries/simpleCUBLASXT"
    "${BENCHMARK_SAMPLES_ROOT}/4_CUDA_Libraries/conjugateGradientMultiDeviceCG"
    "${BENCHMARK_SAMPLES_ROOT}/5_Domain_Specific/MonteCarloMultiGPU"
)

configure_and_build() {
    local src_dir="$1"
    local sample_name="${src_dir##*/}"
    local build_dir="$BUILD_ROOT/$sample_name"

    echo "==> Configuring ${sample_name}"
    cmake -S "$src_dir" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=90 \
        -DCUDAToolkit_ROOT="$CUDA_DIR" \
        -DCMAKE_CUDA_COMPILER="$CUDA_DIR/bin/nvcc"

    echo "==> Building ${sample_name}"
    cmake --build "$build_dir" --parallel "$JOBS" --target "$sample_name"
}

for sample in "${SAMPLES[@]}"; do
    configure_and_build "$sample"
done

echo "Built samples into $BUILD_ROOT"
