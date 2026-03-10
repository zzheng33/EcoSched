#!/bin/bash

python3 /home/ac.zzheng/power/GPGPU/script/co-sched/tenant.py \
  --app resnet50 --gpus 0,1 --numa 0 --batch-size 2048