#!/bin/bash

# 脚本用于运行 Torchvision 中所有分类模型的训练
# 数据集目录由环境变量指定：
#   - DATA_DIR: 必须指定的数据集目录 (包含训练数据)

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3

# 读取环境变量
DATA_DIR=${DATA_DIR:-""}

# Help message
usage() {
    echo "Usage: DATA_DIR=<dataset_path> $0"
    echo "  - DATA_DIR (required)      Path to the dataset directory"
    echo "  - Example: DATA_DIR=/path/to/imagenet2012 $0"
    exit 1
}

# 检查环境变量是否已设置
if [ -z "$DATA_DIR" ]; then
    echo "Error: DATA_DIR (dataset path) is required."
    usage
fi

# 确保数据集路径存在
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Dataset directory '$DATA_DIR' does not exist."
    exit 1
fi

# 模型列表
models=(
    alexnet convnext_base convnext_large convnext_small convnext_tiny
    densenet121 densenet161 densenet169 densenet201
    efficientnet_b0 efficientnet_b1 efficientnet_b2 efficientnet_b3
    efficientnet_b4 efficientnet_b5 efficientnet_b6 efficientnet_b7
    googlenet inception_v3 mnasnet0_5 mnasnet0_75 mnasnet1_0 mnasnet1_3
    mobilenet_v2 mobilenet_v3_large mobilenet_v3_small
    regnet_x_16gf regnet_x_1_6gf regnet_x_32gf regnet_x_3_2gf
    regnet_x_400mf regnet_x_800mf regnet_x_8gf regnet_y_128gf
    regnet_y_16gf regnet_y_1_6gf regnet_y_32gf regnet_y_3_2gf
    regnet_y_400mf regnet_y_800mf regnet_y_8gf resnet101 resnet152
    resnet18 resnet34 resnet50 resnext101_32x8d resnext50_32x4d
    shufflenet_v2_x0_5 shufflenet_v2_x1_0 shufflenet_v2_x1_5 shufflenet_v2_x2_0
    squeezenet1_0 squeezenet1_1 vgg11 vgg11_bn vgg13 vgg13_bn vgg16
    vgg16_bn vgg19 vgg19_bn vit_b_16 vit_b_32 vit_l_16 vit_l_32
    wide_resnet101_2 wide_resnet50_2
)

echo "Training start: $(date +'%m/%d/%Y %T')"

# 遍历所有模型
for model in "${models[@]}"; do
    echo "Training $model start: $(date +'%m/%d/%Y %T')"
    
    python main.py \
        -a "$model" \
        --dist-backend 'nccl' \
        --dist-url "tcp://localhost:8828" \
        --multiprocessing-distributed \
        --world-size 1 \
        --rank 0 \
        --batch-size 64 \
        $DATA_DIR &

    # 获取进程 ID
    pid=$!

    # 运行 10 分钟
    sleep 600 

    # 终止训练进程及其子进程
    echo "Stopping training process (PID: $pid)..."
    pkill -P "$pid" || true
    kill "$pid" 2>/dev/null || true

    echo "Training $model finish: $(date +'%m/%d/%Y %T')"

    # 等待缓冲区刷新
    sleep 20
done
