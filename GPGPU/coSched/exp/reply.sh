#!/bin/bash



# python3 replay_schedule.py \
#     --log results/H100/mix/EcoPack_cmab_run.txt \
#     --results-dir results/H100/mix \



# Solo runs (same GPU/NUMA binding as in EcoPack)

# python3 replay_schedule.py \
#     --log results/H100/mix/EcoPack_cmab_run.txt \
#     --solo simpleP2P \
#     --no-log \
#     --gpus 2,3 --numa 1

# python3 replay_schedule.py \
#     --log results/H100/mix/EcoPack_cmab_run.txt \
#     --solo MonteCarloMultiGPU \
#     --no-log \
#     --gpus 2 --numa 1

# python3 replay_schedule.py \
#     --log results/H100/mix/EcoPack_cmab_run.txt \
#     --solo vgg16 \
#     --no-log \
#     --gpus 0 --numa 0

# python3 replay_schedule.py \
#     --log results/H100/mix/EcoPack_cmab_run.txt \
#     --solo vgg19 \
#     --no-log \
#     --gpus 2 --numa 1

# Co-run pairs

# python3 replay_schedule.py \
#     --log results/H100/mix/EcoPack_cmab_run.txt \
#     --pair simpleP2P pot3d \
#     --no-log

python3 replay_schedule.py \
    --log ../results/H100/mix/EcoPack_cmab_run.txt \
    --pair MonteCarloMultiGPU pot3d \
    --no-log

# python3 replay_schedule.py \
#     --log results/H100/mix/EcoPack_cmab_run.txt \
#     --pair vgg16 cloverleaf \
#     --no-log

# python3 replay_schedule.py \
#     --log results/H100/mix/EcoPack_cmab_run.txt \
#     --pair vgg19 cloverleaf \
#     --results-dir results/H100/mix \
#     --no-log



# simpleP2P solo — 2 GPUs [2,3], NUMA 1
# time CUDA_VISIBLE_DEVICES=2,3 numactl --cpunodebind=1 --membind=1 bash ~/power/GPGPU/script/run_benchmark/cuda_script/simpleP2P.sh 2

# # MonteCarloMultiGPU solo — 1 GPU [2], NUMA 1
# time CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=1 --membind=1 bash ~/power/GPGPU/script/run_benchmark/cuda_script/MonteCarloMultiGPU.sh 1

# # vgg16 solo — 1 GPU [2], NUMA 1
# time CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=1 --membind=1 python3 ~/power/ML/dl.py --model vgg16 --num-gpus 1 --batch-size 2048 --epochs 1 --lr 0.1

# # vgg19 solo — 1 GPU [2], NUMA 1
# time CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=1 --membind=1 python3 ~/power/ML/dl.py --model vgg19 --num-gpus 1 --batch-size 2048 --epochs 1 --lr 0.1

# # pot3d solo — 2 GPUs [0,1], NUMA 0
# time CUDA_VISIBLE_DEVICES=0,1 numactl --cpunodebind=0 --membind=0 bash ~/power/GPGPU/script/run_benchmark/spec_script/pot3d.sh 2

# # cloverleaf solo — 3 GPUs [0,1,3], NUMA 0
# time CUDA_VISIBLE_DEVICES=0,1,3 numactl --cpunodebind=0 --membind=0 bash ~/power/GPGPU/script/run_benchmark/spec_script/cloverleaf.sh 3
