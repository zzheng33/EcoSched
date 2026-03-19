#!/bin/bash

# HPGMG Benchmark Run Script
# Usage: ./run_hpgmg.sh [MPI_RANKS] [LOG2_BOX_DIM] [BOXES_PER_RANK]
# Example: ./run_hpgmg.sh 4 9 2

MPI_RANKS=${1:-4}         # Default to 4 GPUs if not specified
LOG2_BOX_DIM=${2:-8}      # Default to 512³ boxes (2^9)
BOXES_PER_RANK=${3:-8}    # Default to 2 boxes per rank

cd ${SPEC_BENCHMARK_ROOT:-/home/ac.zzheng/benchmark/spec}/hpgmg

# Run HPGMG
# If CUDA_VISIBLE_DEVICES is set, pick the Nth GPU from that list by MPI rank.
# Otherwise fall back to using MPI local rank as the GPU id.
if [ "${MPI_RANKS}" -eq 1 ]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS="," read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
        export CUDA_VISIBLE_DEVICES=${GPULIST[0]}
    else
        export CUDA_VISIBLE_DEVICES=0
    fi
    ./build/bin/hpgmg-fv ${LOG2_BOX_DIM} ${BOXES_PER_RANK}
else
    mpirun -np ${MPI_RANKS} bash -c '
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS="," read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
        export CUDA_VISIBLE_DEVICES=${GPULIST[$OMPI_COMM_WORLD_LOCAL_RANK]}
    else
        export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
    fi
    ./build/bin/hpgmg-fv '"${LOG2_BOX_DIM}"' '"${BOXES_PER_RANK}"''
fi
