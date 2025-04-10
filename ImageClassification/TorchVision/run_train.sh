#!/bin/bash

# 脚本用于运行 Torchvision 中所有分类模型的训练
# 数据集目录和模型架构均由环境变量指定：
#   - ARCH: 必须指定的模型架构 (自动转换为小写)
#   - DATA_DIR: 必须指定的数据集目录

set -e

export MLU_VISIBLE_DEVICES=${MLU_VISIBLE_DEVICES:-"0,1,2,3"}

# 设置默认MLU设备
# 读取环境变量，并将 ARCH 转换为小写
ARCH=${ARCH:-""}
ARCH=$(echo "$ARCH" | tr '[:upper:]' '[:lower:]')  # 转换为小写
DATA_DIR=${DATA_DIR:-""}

# Help message
usage() {
    echo "Usage: ARCH=<model> DATA_DIR=<dataset_path> $0"
    echo "  - ARCH (required)          Model architecture (e.g., alexnet, resnet18, etc.)"
    echo "  - MLU_VISIBLE_DEVICES      Comma-separated MLU device IDs (default: 0,1,2,3)"
    echo "  - DATA_DIR (required)      Path to the dataset directory"
    echo "  - Example: ARCH=ResNet18 DATA_DIR=/path/to/imagenet2012 $0"
    exit 1
}

# 检查必要参数
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

echo "=== MLU Configuration ==="
echo "  - Using MLU Devices: $MLU_VISIBLE_DEVICES"
echo "  - Model Architecture: $ARCH"
echo "  - Dataset Path: $DATA_DIR"
echo "========================="

#训练执行
echo "Training Start: $(date +'%m/%d/%Y %T')" 

# 4 card training
echo "Training $ARCH..."
python main.py \
    -a "$ARCH" \
    --device 'mlu' \
    --dist-backend 'cncl' \
    --dist-url "tcp://localhost:8828" \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --batch-size 64 \
    --workers 0 \
    --cnmix \
    $DATA_DIR

echo "Training Finish: $(date +'%m/%d/%Y %T')"

