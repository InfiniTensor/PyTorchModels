#!/bin/bash

# 确保脚本在遇到错误时停止执行
set -e

export MLU_VISIBLE_DEVICES=0  

# 数据集设置
DATASET_ROOT="/dataset/VOC2007"
[ ! -d "$DATASET_ROOT/VOCdevkit" ] && {
    echo "创建数据集符号链接..."
    mkdir -p $(dirname "$DATASET_ROOT")
    ln -sf /data1/shared/Dataset/VOCdevkit "$DATASET_ROOT/VOCdevkit"
}

# 运行训练脚本
python train.py \
    --device mlu \
    --batch_size 4 \
    --epochs 10 \
    --input_size 256 \
    --classes 21 \
    --dataset_path "$DATASET_ROOT" \
    --VOC_year 2007 \
    --saved_dir "./saved_models"
