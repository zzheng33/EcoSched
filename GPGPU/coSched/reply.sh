#!/bin/bash



# python3 replay_schedule.py \
#     --log results/H100/mix/EcoPack_cmab_run.txt \
#     --results-dir results/H100/mix \



# Solo runs (same GPU/NUMA binding as in EcoPack)
python3 replay_schedule.py \
    --log results/H100/mix/EcoPack_cmab_run.txt \
    --solo simpleP2P \
    --results-dir results/H100/mix

python3 replay_schedule.py \
    --log results/H100/mix/EcoPack_cmab_run.txt \
    --solo MonteCarloMultiGPU \
    --results-dir results/H100/mix

python3 replay_schedule.py \
    --log results/H100/mix/EcoPack_cmab_run.txt \
    --solo vgg16 \
    --results-dir results/H100/mix

python3 replay_schedule.py \
    --log results/H100/mix/EcoPack_cmab_run.txt \
    --solo vgg19 \
    --results-dir results/H100/mix

# Co-run pairs
python3 replay_schedule.py \
    --log results/H100/mix/EcoPack_cmab_run.txt \
    --pair simpleP2P pot3d \
    --results-dir results/H100/mix

python3 replay_schedule.py \
    --log results/H100/mix/EcoPack_cmab_run.txt \
    --pair MonteCarloMultiGPU pot3d \
    --results-dir results/H100/mix

python3 replay_schedule.py \
    --log results/H100/mix/EcoPack_cmab_run.txt \
    --pair vgg16 cloverleaf \
    --results-dir results/H100/mix

python3 replay_schedule.py \
    --log results/H100/mix/EcoPack_cmab_run.txt \
    --pair vgg19 cloverleaf \
    --results-dir results/H100/mix
