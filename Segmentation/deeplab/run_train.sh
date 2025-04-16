#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

set -e

export MLU_VISIBLE_DEVICES=0

# 数据集设置
DATASET_ROOT="/dataset/VOC2007"
[ ! -d "$DATASET_ROOT/VOCdevkit" ] && {
    echo "创建数据集符号链接..."
    mkdir -p $(dirname "$DATASET_ROOT")
    ln -sf /data1/shared/Dataset/VOCdevkit "$DATASET_ROOT/VOCdevkit"
}

# 运行训练
python $SCRIPT_DIR/deeplab.py \
    --device mlu \
    --train-batch-size 4 \
    --train-epochs 10 \
    --mode train \
    --image-size 256 \
