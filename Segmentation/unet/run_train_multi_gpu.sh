#!/bin/bash

# 多卡训练脚本
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

if [ -e "../data/VOCdevkit" ]; then
    echo "../data/VOCdevkit exists"
else
    ln -sfn /data-aisoft/Dataset/VOCdevkit ../data/VOCdevkit
fi

torchrun --nproc_per_node=4 train.py \
       --dataset_path ../data \
       --VOC_year 2007 \
       --batch_size 4 \
       --epochs 100 \
       --input_size 256 \
       --classes 21
