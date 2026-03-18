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
BUILD_ROOT="${BUILD_ROOT:-$SCRIPT_DIR/build-sm90}"
SOURCE_ROOT="${SOURCE_ROOT:-$BUILD_ROOT/source}"
JOBS="${JOBS:-$(nproc)}"

SAMPLES=(
    "Samples/4_CUDA_Libraries/simpleCUFFT_2d_MGPU"
    "Samples/4_CUDA_Libraries/simpleCUFFT_MGPU"
    "Samples/4_CUDA_Libraries/simpleCUBLASXT"
    "Samples/4_CUDA_Libraries/conjugateGradientMultiDeviceCG"
    "Samples/5_Domain_Specific/MonteCarloMultiGPU"
)

reset_build_dir_if_needed() {
    local src_dir="$1"
    local build_dir="$2"
    local cache_file="$build_dir/CMakeCache.txt"

    if [[ -f "$cache_file" ]]; then
        local cached_source
        cached_source=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$cache_file")

        if [[ -n "$cached_source" && "$cached_source" != "$src_dir" ]]; then
            echo "==> Resetting $build_dir because cached source is $cached_source"
            python3 - "$build_dir" <<'INNERPY'
import shutil
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.exists():
    shutil.rmtree(path)
INNERPY
        fi
    fi
}

configure_and_build() {
    local rel_path="$1"
    local sample_name="${rel_path##*/}"
    local src_dir="$SOURCE_ROOT/$rel_path"
    local build_dir="$BUILD_ROOT/$sample_name"

    reset_build_dir_if_needed "$src_dir" "$build_dir"

    echo "==> Configuring ${sample_name} from ${src_dir}"
    cmake -S "$src_dir" -B "$build_dir"         -DCMAKE_BUILD_TYPE=Release         -DCMAKE_CUDA_ARCHITECTURES=90         -DCUDAToolkit_ROOT="$CUDA_DIR"         -DCMAKE_CUDA_COMPILER="$CUDA_DIR/bin/nvcc"

    echo "==> Building ${sample_name}"
    cmake --build "$build_dir" --parallel "$JOBS" --target "$sample_name"
}

for sample in "${SAMPLES[@]}"; do
    configure_and_build "$sample"
done

echo "Built samples into $BUILD_ROOT"
echo "Sources at $SOURCE_ROOT"
