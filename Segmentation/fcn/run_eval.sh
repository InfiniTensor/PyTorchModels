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

python $SCRIPT_DIR/fcn.py \
	--device mlu \
        --infer-batch-size 1 \
        --image-size 256 \
        --mode infer \
