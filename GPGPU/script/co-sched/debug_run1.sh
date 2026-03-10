#!/bin/bash
# Direct resnet50 on GPU 0,1,2, pinned to NUMA 0 CPUs
export CUDA_VISIBLE_DEVICES=0,1,2
cd ~/power/ML
numactl --physcpubind=0-47,96-143 --membind=0 \
  ~/env/ml/bin/python3 dl.py --model resnet50 --num-gpus 3 --batch-size 2048 --epochs 100 --lr 0.001