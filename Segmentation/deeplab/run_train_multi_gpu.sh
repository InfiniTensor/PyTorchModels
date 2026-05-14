#!/bin/bash

# 多卡训练脚本
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3

if [ -e "../data/VOCdevkit" ]; then
    echo "../data/VOCdevkit exists"
else
    ln -s /data1/shared/Dataset/VOCdevkit ../data/VOCdevkit
fi

torchrun --nproc_per_node=4 $SCRIPT_DIR/deeplab.py \
        --train-batch-size 4 \
        --train-epochs 10 \
        --mode train \
        --image-size 256
