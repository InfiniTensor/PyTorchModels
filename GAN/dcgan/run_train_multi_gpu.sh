#!/bin/bash
set -e

# 多卡训练脚本
DATA_DIR=${DATA_DIR:-""}
OUTPUT_DIR="${OUTPUT_DIR:-./output}"

# Help message
usage() {
    echo "Usage: DATA_DIR=<dataset_path> $0"
    echo "  - DATA_DIR (required)      Path to the dataset directory (should contain 'val' subdirectory)"
    echo "  - OUTPUT_DIR (optional)      Path to the output directory"
	echo "  - Example: DATA_DIR=/path/to/lsun OUTPUT_DIR=./output $0"
    exit 1
}

# 确保数据集路径存在
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Dataset directory '$DATA_DIR' does not exist."
    exit 1
fi

# Process output directory
echo "Cleaning up the old output directory..."
rm -rf "$OUTPUT_DIR"

echo "Creating a new output directory..."
mkdir "$OUTPUT_DIR"

export MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0,1}
PYTHONUNBUFFERED=1 torchrun --nproc_per_node=4 train.py --dataset fake --cuda
