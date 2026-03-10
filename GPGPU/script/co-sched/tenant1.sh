#!/bin/bash
# Tenant 1: NUMA 0 (CPUs 0-47,96-143)
# Usage: bash run1.sh --app resnet50 --gpus 0,1,2 [--batch-size 2048] [--epochs 100] [--lr 0.001]
#        bash run1.sh --app gups --gpus 0
#        bash run1.sh --app bert --gpus 0,1

set -euo pipefail

CPUBIND="0-47,96-143"
MEMBIND=0

HOME_DIR="$HOME"
BENCH_ROOT="$HOME_DIR/power/GPGPU/script/run_benchmark"
ML_PYTHON="$HOME_DIR/env/ml/bin/python3"
ML_SCRIPT="$HOME_DIR/power/ML/dl.py"
ML_WORKDIR="$HOME_DIR/power/ML"

# --- Parse arguments ---
APP=""
GPUS=""
BATCH_SIZE=2048
EPOCHS=3
LR=0.001

while [[ $# -gt 0 ]]; do
    case "$1" in
        --app)        APP="$2";        shift 2 ;;
        --gpus)       GPUS="$2";       shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --lr)         LR="$2";         shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$APP" || -z "$GPUS" ]]; then
    echo "Usage: bash run1.sh --app <name> --gpus <id,id,...> [--batch-size N] [--epochs N] [--lr F]"
    exit 1
fi

NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

export CUDA_VISIBLE_DEVICES="$GPUS"
export LD_LIBRARY_PATH="/soft/compilers/cuda/cuda-12.3.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[INFO] Tenant 1 | App: $APP | GPUs: $GPUS | NUMA: $MEMBIND | CPUs: $CPUBIND"

# --- Resolve and run ---

# DL models: resnet50, vgg16
if [[ "$APP" == "resnet50" || "$APP" == "vgg16" ]]; then
    cd "$ML_WORKDIR"
    exec numactl --physcpubind="$CPUBIND" --membind="$MEMBIND" \
        "$ML_PYTHON" "$ML_SCRIPT" \
        --model "$APP" --num-gpus "$NUM_GPUS" \
        --batch-size "$BATCH_SIZE" --epochs "$EPOCHS" --lr "$LR"
fi

# SPEC benchmarks (need module environment)
declare -a SPEC_APPS=(lbm cloverleaf tealeaf minisweep pot3d miniweather hpgmg)
for s in "${SPEC_APPS[@]}"; do
    if [[ "$APP" == "$s" ]]; then
        source /etc/profile >/dev/null 2>&1 || true
        source /etc/profile.d/modules.sh >/dev/null 2>&1 || true
        module use /soft/modulefiles
        module load cuda/12.3.0
        module load cmake
        module load gcc/12.2.0
        module load openmpi/4.1.1-gcc
        module load public_mkl/2019
        export CUDA_DIR=/soft/compilers/cuda/cuda-12.3.0
        export PCM_NO_MSR=1
        export PCM_KEEP_NMI_WATCHDOG=1
        # Re-export after module loads (modules may override)
        export CUDA_VISIBLE_DEVICES="$GPUS"
        exec numactl --physcpubind="$CPUBIND" --membind="$MEMBIND" \
            bash "$BENCH_ROOT/spec_script/${APP}.sh" "$NUM_GPUS"
    fi
done

# ALTIS benchmarks
declare -A ALTIS_LEVEL=(
    [maxflops]=level0
    [bfs]=level1 [gemm]=level1 [gups]=level1 [pathfinder]=level1 [sort]=level1
    [cfd]=level2 [cfd_double]=level2 [fdtd2d]=level2 [kmeans]=level2 [lavamd]=level2
    [nw]=level2 [particlefilter_float]=level2 [particlefilter_naive]=level2
    [raytracing]=level2 [srad]=level2 [where]=level2
)
if [[ -v "ALTIS_LEVEL[$APP]" ]]; then
    exec numactl --physcpubind="$CPUBIND" --membind="$MEMBIND" \
        bash "$BENCH_ROOT/altis_script/${ALTIS_LEVEL[$APP]}/${APP}.sh"
fi

# ECP benchmarks (bert, gpt2, and others)
declare -a ECP_APPS=(XSBench miniGAN CRADL sw4lite Laghos bert UNet Resnet50 lammps gromacs NAMD gpt2)
for s in "${ECP_APPS[@]}"; do
    if [[ "$APP" == "$s" ]]; then
        exec numactl --physcpubind="$CPUBIND" --membind="$MEMBIND" \
            bash "$BENCH_ROOT/ecp_script/${APP}.sh" "$NUM_GPUS"
    fi
done

# HEC benchmarks
declare -a HEC_APPS=(addBiasResidualLayerNorm aobench background-subtract chacha20 convolution3D dropout extrema fft kalman knn softmax stencil3d zmddft zoom)
for s in "${HEC_APPS[@]}"; do
    if [[ "$APP" == "$s" ]]; then
        exec numactl --physcpubind="$CPUBIND" --membind="$MEMBIND" \
            bash "$BENCH_ROOT/hec_script/${APP}.sh"
    fi
done

echo "[ERROR] Unknown app: $APP"
exit 1
