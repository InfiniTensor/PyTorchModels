#!/bin/bash

# 脚本用于运行 YOLOv5 评估
# 依赖以下环境变量：
#   - MODEL: 必须指定的模型架构 (自动转换为小写)
#   - DATA_DIR: 必须指定的数据集目录

set -e

export CUDA_VISIBLE_DEVICES=0

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

# 模型权重下载地址
declare -A MODELS
MODELS["yolov5n"]="https://cloud.tsinghua.edu.cn/seafhttp/files/ed37cef3-e656-4bbe-868d-b640ca9645f1/yolov5n.pt"
MODELS["yolov5s"]="https://cloud.tsinghua.edu.cn/seafhttp/files/6739c917-1494-4c26-93bd-4f7d50f02f2e/yolov5s.pt"
MODELS["yolov5m"]="https://cloud.tsinghua.edu.cn/seafhttp/files/3f3a1cf9-3e74-43bd-9ed1-dfd9396246c6/yolov5m.pt"
MODELS["yolov5l"]="https://cloud.tsinghua.edu.cn/seafhttp/files/91ba88be-99b8-47fd-8c34-8914f66d840e/yolov5l.pt"
MODELS["yolov5x"]="https://cloud.tsinghua.edu.cn/seafhttp/files/927abf57-76c8-4f27-97a7-948e9d2f7f90/yolov5x.pt"

# 检查模型是否有效
if [[ -z "${MODELS[$MODEL]}" ]]; then
    echo "Error: Invalid model selection. Choose from yolov5n, yolov5s, yolov5m, yolov5l, yolov5x."
    exit 1
fi

MODEL_PATH="./${MODEL}.pt"

# 下载模型权重
if [ -e "$MODEL_PATH" ]; then
    echo "$MODEL_PATH exists"
else
    echo "Downloading $MODEL from ${MODELS[$MODEL]}"
    wget -O "$MODEL_PATH" "${MODELS[$MODEL]}"
fi

echo "Evaluation Start: $(date +'%m/%d/%Y %T')"

echo "Evaluating $MODEL..."
if [ -f "$MODEL_PATH" ]; then
    python3 val.py --weights "$MODEL_PATH" --data coco.yaml --img 640
else
    echo "Error: Model path $MODEL_PATH does not exist."
    exit 1
fi

echo "Evaluation Finish: $(date +'%m/%d/%Y %T')"

# mAP50: 0.565 mAP50-95: 0.371
