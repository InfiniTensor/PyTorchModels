#!/bin/bash

set -e

# 多卡训练脚本
export CUDA_VISIBLE_DEVICES=0,1,2,3

saved_dir=$1
num_episodes=$2
lr=$3

torchrun --nproc_per_node=4 main.py \
       --save_path $saved_dir \
       --num_episodes $num_episodes \
       --lr=$lr

# bash run_train_multi_gpu.sh checkpoints 100 0.0001
