#!/bin/bash

# 脚本用于训练和预测模型
# 必须传递的参数：
#   - PYTORCH_INFER_CHECKPOINT: 必须指定的模型检查点路径
#   - PYTORCH_TRAIN_DATASET: 必须指定的数据集根目录
# 脚本会执行模型的训练，并在指定目录下保存相关文件。

set -e  # 一旦遇到错误，退出脚本

# 读取命令行参数
PYTORCH_INFER_CHECKPOINT=${MODEL:-""}
PYTORCH_TRAIN_DATASET=${DATA_DIR:-""}

# 检查参数是否传递
if [ -z "$PYTORCH_INFER_CHECKPOINT" ]; then
    echo "Error: PYTORCH_INFER_CHECKPOINT is required (e.g., model checkpoint path)"
    exit 1
fi

if [ -z "$PYTORCH_TRAIN_DATASET" ]; then
    echo "Error: PYTORCH_TRAIN_DATASET is required (e.g., dataset root path)"
    exit 1
fi

# 获取当前脚本所在目录
CUR_DIR=$(cd $(dirname $0); pwd)

# 定义缓存目录
cache_dir=${CUR_DIR}/data

# 数据集目录
DATASET_DIR=${PYTORCH_TRAIN_DATASET}/ml-20mx4x16

# 检查数据集目录是否存在
if [ ! -d "${DATASET_DIR}" ]; then
    echo "Error: Directory ${DATASET_DIR} does not exist"
    exit 1
fi

# 开始训练
echo "Starting training with checkpoint ${PYTORCH_INFER_CHECKPOINT} and dataset ${DATASET_DIR}"

python ncf.py \
    --data ${DATASET_DIR} \
    --resume ${PYTORCH_INFER_CHECKPOINT} \
    -l 0.0002 \
    -b 65536 \
    --layers 256 256 128 64 \
    -f 64 \
    --seed 0 \
    --save_ckp 1 \
    --threshold 1.0 \
    --user_scaling 4 \
    --item_scaling 16 \
    --cpu_dataloader \
    --random_negatives \
    --device gpu \
    --workers 8 \
    --do_predict \
    --cachedir ${cache_dir} \
    --multiprocessing-distributed

echo "Training finished successfully"

