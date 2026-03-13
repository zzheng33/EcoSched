#!/bin/bash
# Direct raytracing on GPU 3, pinned to NUMA 1 CPUs
export CUDA_VISIBLE_DEVICES=3
export LD_LIBRARY_PATH=/soft/compilers/cuda/cuda-12.3.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

numactl --physcpubind=48-95,144-191 --membind=1 \
 bash ../run_benchmark/altis_script/level1/gups.sh