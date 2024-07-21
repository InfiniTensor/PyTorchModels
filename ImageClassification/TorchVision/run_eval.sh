#!/bin/bash

# 脚本用于运行 Torchvision 中所有分类模型的的训练
# 数据集放在 /data1/shared/Dataset/imagenet2012

set -e

export CUDA_VISIBLE_DEVICES=0

ARCH=""
LOG_FILE=""

# Help message
usage() {
    echo "Usage: $0 -a ARCH"
    echo "  -a ARCH, --arch ARCH       Model architecture (e.g., AlexNet, ResNet18, etc.)"
    echo "  -h, --help                 Display this help and exit"
    echo "  -l LOG_FILE, --log LOG_FILE Log file"
    exit 1
}

# Parse command line arguments
while getopts "a:l:h" opt; do
    case $opt in
        a) ARCH=$OPTARG ;;
        l) LOG_FILE=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Check ARCH parsing
if [ -z "$ARCH" ]; then
    echo "Error: Model architecture is required."
    usage
fi

# 检查软连接是否已经存在了
if [ -e "../data/imagenet2012" ]; then
    echo "../data/imagenet2012 exists"
else
    # 创建软连接
    ln -s /data1/shared/Dataset/imagenet2012 ../data
fi

# Define log file with ARCH included
if [ -z "$LOG_FILE" ]; then
    LOG_FILE="pytorch-${ARCH}-eval-gpu${CUDA_VISIBLE_DEVICES}.log"
    echo "Logs saved in defaut log file ${LOG_FILE}"
else 
    echo "Logs saved in ${LOG_FILE}"
fi

echo "Evaluating Start: $(date +'%m/%d/%Y %T')" > ${LOG_FILE}

# 单机单卡推理
echo "Evaluating $ARCH..."
python main.py \
    -a $ARCH \
    --world-size 1 \
    --batch-size 64 \
    --pretrained \
    --evaluate \
    ../data/imagenet2012 2>&1 | tee -a $LOG_FILE

echo "Evaluating Finish: $(date +'%m/%d/%Y %T')" >> ${LOG_FILE}
