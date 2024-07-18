#!/bin/bash

# 这个脚本用于运行Python脚本
# 数据集放在同级目录下  ./data

# 确保脚本在遇到错误时停止执行
set -e

export CUDA_VISIBLE_DEVICES=0

MODEL_DIR=./model
DATASETS=(
    "jenaclimate",
    "electricity",
    "airpassengers"
)

dataset=$1

# 检查输入是否有效  
is_valid=false  
for data in "${DATASETS[@]}"; do  
    if [[ "$dataset" == "$data" ]]; then  
        is_valid=true  
        break  
    fi  
done 

python lstm.py --mode train \
               --seq_length 24 \
               --batch_size 16 \
               --train_eval_split 0.8 \
               --dataset $dataset \
               --epochs 4 \
               --data ../data \