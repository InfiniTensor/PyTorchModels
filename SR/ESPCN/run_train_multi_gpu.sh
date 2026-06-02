#!/bin/bash

# 多卡训练脚本
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

python data_utils.py --upscale_factor 2

torchrun --nproc_per_node=4 train.py --upscale_factor 2 --num_epochs 2
