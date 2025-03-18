#!/bin/bash

# 脚本用于运行 Torchvision 中所有分类模型的推理
# 数据集目录和模型架构均由环境变量指定：
#   - ARCH: 必须指定的模型架构 (自动转换为小写)
#   - DATA_DIR: 必须指定的数据集目录

set -e

export CUDA_VISIBLE_DEVICES=0

# 读取环境变量，并将 ARCH 转换为小写
ARCH=${ARCH:-""}
ARCH=$(echo "$ARCH" | tr '[:upper:]' '[:lower:]')  # 转换为小写
DATA_DIR=${DATA_DIR:-""}

# Help message
usage() {
    echo "Usage: ARCH=<model> DATA_DIR=<dataset_path> $0"
    echo "  - ARCH (required)          Model architecture (e.g., alexnet, resnet18, etc.)"
    echo "  - DATA_DIR (required)      Path to the dataset directory"
    echo "  - Example: ARCH=ResNet18 DATA_DIR=/path/to/imagenet2012 $0"
    exit 1
}

# 检查环境变量是否已设置
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

# 软链接数据目录
if [ -e "../data/imagenet2012" ]; then
    echo "Dataset ../data/imagenet2012 exists"
else
    echo "Linking dataset from $DATA_DIR to ../data/imagenet2012"
    ln -s "$DATA_DIR" ../data
fi

echo "Evaluating Start: $(date +'%m/%d/%Y %T')"

# 单机单卡推理
echo "Evaluating $ARCH..."
python main.py \
    -a "$ARCH" \
    --world-size 1 \
    --batch-size 64 \
    --pretrained \
    --evaluate \
    ../data/imagenet2012

echo "Evaluating Finish: $(date +'%m/%d/%Y %T')"
