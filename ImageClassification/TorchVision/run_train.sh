#!/bin/bash

# 脚本用于运行 Torchvision 中所有分类模型的的训练
# 数据集放在 /data1/shared/Dataset/imagenet2012

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3

ARCH=""

# Help message
usage() {
    echo "Usage: $0 -a ARCH"
    echo "  -a ARCH, --arch ARCH       Model architecture (e.g., AlexNet, ResNet18, etc.)"
    echo "  -h, --help                 Display this help and exit"
    exit 1
}

# Parse command line arguments
while getopts "a:h" opt; do
    case $opt in
        a) ARCH=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Check ARCH parsing
if [ -z "$ARCH" ]; then
    echo "Error: Model architecture is required."
    usage
fi

# Check dataset
if [ -e "../data/imagenet2012" ]; then
    echo "Dataset ../data/imagenet2012 exists"
else
    # Link defaut dataset
    ln -s /data1/shared/Dataset/imagenet2012 ../data
fi

echo "Training Start: $(date +'%m/%d/%Y %T')" 

# 4 card training
echo "Training $ARCH..."
python main.py \
    -a $ARCH \
    --dist-backend 'nccl' \
    --dist-url "tcp://localhost:8828" \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --batch-size 64 \
    ../data/imagenet2012

echo "Training Finish: $(date +'%m/%d/%Y %T')"
