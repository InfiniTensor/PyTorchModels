#!/bin/bash

# 多卡训练脚本
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

dataset=$1
epoch=$2
batch_size=$3
learning_rate=$4

torchrun --nproc_per_node=4 -W ignore train.py \
       --dataset $dataset \
       --epoch $epoch \
       --batch_size $batch_size \
       --lr $learning_rate

# bash run_train_multi_gpu.sh ../data/complete_data.csv 200 512 0.0001
