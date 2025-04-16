#!/bin/bash

# 脚本用于运行 YOLOv5 训练
# 依赖以下环境变量：
#   - MODEL: 必须指定的模型架构 (自动转换为小写)
#   - DATA_DIR: 必须指定的数据集目录

set -e

export ASCEND_RT_VISIBLE_DEVICES=1,2,3

# 读取环境变量，并将 MODEL 转换为小写
MODEL=${MODEL:-"yolov5s"}
MODEL=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')  # 转换为小写
DATA_DIR=${DATA_DIR:-""}

# Help message
usage() {
    echo "Usage: MODEL=<model> DATA_DIR=<dataset_path> $0"
    echo "  - MODEL (required)         YOLOv5 model variant (e.g., yolov5n, yolov5s, etc.)"
    echo "  - DATA_DIR (required)      Path to the dataset directory"
    echo "  - Example: MODEL=yolov5s DATA_DIR=/path/to/coco $0"
    exit 1
}

# 检查环境变量是否已设置
if [ -z "$MODEL" ]; then
    echo "Error: MODEL (YOLOv5 model variant) is required."
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
if [ -e "../data/coco" ]; then
    echo "Dataset ../data/coco exists"
else
    echo "Linking dataset from $DATA_DIR to ../data/coco"
    ln -s "$DATA_DIR" ../data
fi

# 复制字体文件
mkdir -p ~/.config/Ultralytics/
cp ./Arial.ttf ~/.config/Ultralytics/Arial.ttf

echo "Training Start: $(date +'%m/%d/%Y %T')"

# 运行 YOLOv5 训练
echo "Training $MODEL..."
python -m torch.distributed.run \
    --nproc_per_node 2 train.py \
    --batch 64 \
    --img 640 \
    --epoch 25 \
    --data coco.yaml \
    --weights "" \
    --cfg "models/${MODEL}.yaml" \
    --device 0,1 \
    --nosave \
    --noval \
    --workers 4

echo "Training Finish: $(date +'%m/%d/%Y %T')"

