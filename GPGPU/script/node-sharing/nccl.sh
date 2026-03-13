#!/bin/bash

source /etc/profile.d/modules.sh
module use /soft/modulefiles
module load cuda/12.3.0
module load gcc/12.2.0
module load openmpi/4.1.1-gcc

# cd ~/benchmark/nccl-tests
# make MPI=1 \
#   CUDA_HOME=/soft/compilers/cuda/cuda-12.3.0 \
#   NCCL_HOME=~/env/ml/lib/python3.11/site-packages/nvidia/nccl \
#   MPI_HOME=$(dirname $(dirname $(which mpirun)))


export CUDA_VISIBLE_DEVICES=0,1,2
export LD_LIBRARY_PATH=/soft/compilers/cuda/cuda-12.3.0/lib64:/soft/libraries/mpi/openmpi/4.1.1/lib:~/env/ml/lib/python3.11/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
cd /home/ac.zzheng/benchmark/nccl-tests/
./build/all_reduce_perf -b 64G -e 64G -f 2 -g 3 -n 1000

# mpirun -np 3 ./build/all_reduce_perf -b 8G -e 16G -f 2 -g 1 -n 1000

