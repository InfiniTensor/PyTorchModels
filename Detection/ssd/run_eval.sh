#!/bin/bash

# 这个脚本用于运行 Python 脚本
# 数据集路径和检查点路径通过环境变量传入

# 确保脚本在遇到错误时停止执行
set -e

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0

# 获取环境变量，并将 ARCH 转换为小写
DATA_DIR=${DATA_DIR:-""}
CKPT_PATH=${CKPT_PATH:-"./checkpoint_ssd300.pth.tar"}
CKPT_URL=${CKPT_URL:-"https://cloud.tsinghua.edu.cn/seafhttp/files/9d78bd21-6e46-4677-b825-f03af94cec00/checkpoint_ssd300.pth.tar"}

# 帮助信息
usage() {
    echo "Usage: ARCH=<model> DATA_DIR=<dataset_path> $0"
    echo "  - ARCH (required)          Model architecture (e.g., ssd300, resnet, etc.)"
    echo "  - DATA_DIR (required)      Path to the dataset directory"
    echo "  - Example: ARCH=ssd300 DATA_DIR=/path/to/dataset $0"
    exit 1
}

# 确保数据集路径存在
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Dataset directory '$DATA_DIR' does not exist."
    exit 1
fi

# 检查模型检查点文件是否存在
if [ -e "$CKPT_PATH" ]; then
    echo "$CKPT_PATH exists"
else
    echo "$CKPT_PATH does not exist, downloading from $CKPT_URL"
    # 下载模型检查点文件
    wget "$CKPT_URL" -O "$CKPT_PATH"
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


# 执行 eval.py 进行评估
echo "Evaluate SSD START"
python eval.py --checkpoint "$CKPT_PATH"

echo "Evaluate SSD FINISHED"

# mAP 0.771