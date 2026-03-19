#!/bin/bash

# miniWeather Benchmark Run Script
# Usage: ./run_miniweather.sh [MPI_RANKS]
# Example: ./run_miniweather.sh 4

MPI_RANKS=${1:-4}  # Default to 4 GPUs if not specified


# NOTE: Problem size is configured at COMPILE TIME
# Current settings (from compile.txt):
#   NX=800        (horizontal grid size)
#   NZ=400        (vertical grid size)
#   SIM_TIME=20000 (simulation time - controls duration)
#
# To change problem size, you need to rebuild:
#   1. Edit compile.txt and modify -DNX, -DNZ, -DSIM_TIME values
#   2. Run the miniWeather compilation section from compile.txt
#
# Larger problem size examples:
#   For 4 GPUs: NX=1600, NZ=800, SIM_TIME=30000
#   For 4 GPUs (huge): NX=3200, NZ=1600, SIM_TIME=40000

cd ${SPEC_BENCHMARK_ROOT:-/home/ac.zzheng/benchmark/spec}/miniWeather/cpp/build

# If CUDA_VISIBLE_DEVICES is set, pick the Nth GPU from that list by MPI rank.
# Otherwise fall back to using MPI local rank as the GPU id.
mpirun -np ${MPI_RANKS} bash -c '
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS="," read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
    export CUDA_VISIBLE_DEVICES=${GPULIST[$OMPI_COMM_WORLD_LOCAL_RANK]}
else
    export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
fi
./parallelfor'
