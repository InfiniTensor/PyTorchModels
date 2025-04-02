#!/bin/bash
set -e

# 使用环境变量，如果没有提供则使用默认路径
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

# 检查并处理输出目录
if [ -e "$output_dir" ]; then
    rm -rf "$OUTPUT_DIR"
else
    mkdir "$OUTPUT_DIR"
fi

export CUDA_VISIBLE_DEVICES=0

python3 train.py --dataset lsun --dataroot "$DATA_DIR" --cuda
