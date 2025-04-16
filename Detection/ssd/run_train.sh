#!/bin/bash

# 这个脚本用于运行 Python 脚本
# 数据集路径通过环境变量传入

# 确保脚本在遇到错误时停止执行
set -e

# 设置 CUDA 设备
export ASCEND_RT_VISIBLE_DEVICES=1,2
# 获取环境变量，并将 ARCH 转换为小写
DATA_DIR=${DATA_DIR:-""}

# 帮助信息
usage() {
    echo "Usage: ARCH=<model> DATA_DIR=<dataset_path> $0"
    echo "  - DATA_DIR (required)      Path to the dataset directory"
    echo "  - Example:DATA_DIR=/path/to/dataset $0"
    exit 1
}

# 确保数据集路径存在
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Dataset directory '$DATA_DIR' does not exist."
    exit 1
fi

# 定义清理数据的函数
cleanup() {
    echo "Cleaning up..."
    rm -rf data/*             # 删除数据目录中的文件
}

# 设置脚本退出时执行清理操作，无论是正常退出还是由于错误中断
trap cleanup EXIT

# 执行数据集预处理
echo "Dataset preprocessing..."
python create_data_lists.py --voc07_path=$DATA_DIR/VOC2007 --voc12_path=$DATA_DIR/VOC2012 --output_folder=./data

# 执行训练
echo "Training SSD START"
python train.py --workers=2

echo "Training SSD FINISHED"
