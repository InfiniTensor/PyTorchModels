#!/bin/bash

# 脚本用于在寒武纪MLU上运行 Torchvision 分类模型推理
# 数据集目录和模型架构由环境变量指定：
#   - ARCH: 必须指定的模型架构 (自动转换为小写)
#   - DATA_DIR: 必须指定的数据集目录

set -e

# 读取并清理环境变量
DATA_DIR=$(echo "$DATA_DIR" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
ARCH=$(echo "$ARCH" | tr '[:upper:]' '[:lower:]' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')

echo "Raw DATA_DIR: '$DATA_DIR'"
DATA_DIR=${DATA_DIR%/}
echo "Processed DATA_DIR: '$DATA_DIR'"

# Help message
usage() {
    echo "Usage: ARCH=<model> DATA_DIR=<dataset_path> $0"
    echo "  - ARCH (required)          Model architecture (e.g., alexnet, resnet18, etc.)"
    echo "  - DATA_DIR (required)      Path to the dataset directory"
    echo "  - Example: ARCH=ResNet50 DATA_DIR=/dataset/ILSVRC2012 $0"
    exit 1
}

# 检查环境变量
if [ -z "$ARCH" ]; then
    echo "Error: ARCH (model architecture) is required."
    usage
fi

if [ -z "$DATA_DIR" ]; then
    echo "Error: DATA_DIR (dataset path) is required."
    usage
fi

# 确保数据集路径存在
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Dataset directory '$DATA_DIR' does not exist."
    exit 1
fi

# 检查数据集结构
if [ ! -d "${DATA_DIR}/val" ] || [ ! -d "${DATA_DIR}/train" ]; then
    echo "ERROR: Dataset structure incorrect:"
    echo "  - Required subdirectories: train/, val/"
    echo "Found contents:"
    ls -l "$DATA_DIR"
    exit 1
fi

echo "Evaluating Start: $(date +'%m/%d/%Y %T')"
echo "Running on MLU with:"
echo "  - ARCH: $ARCH"
echo "  - DATA_DIR: $DATA_DIR"
echo "  - Device: MLU"

# 寒武纪MLU评估命令
python main.py \
    -a "$ARCH" \
    --device mlu \
    --batch-size 64 \
    --pretrained \
    --evaluate \
    --workers 0 \
    "$DATA_DIR"

echo "Evaluating Finish: $(date +'%m/%d/%Y %T')"
