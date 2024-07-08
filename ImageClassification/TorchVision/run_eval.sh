#!/bin/bash

# 脚本用于运行 Torchvision 中所有分类模型的的训练
# 数据集放在 /data1/shared/Dataset/imagenet2012

set -e

# 检查软连接是否已经存在了
if [ -e "../data/imagenet2012" ]; then
    echo "../data/imagenet2012 exists"
else
    # 创建软连接
    ln -s /data1/shared/Dataset/imagenet2012 ../data
fi

model=$1
export CUDA_VISIBLE_DEVICES=0

# 单机单卡推理
python main.py \
    -a $model \
    --world-size 1 \
    --batch-size 64 \
    --pretrained \
    --evaluate \
    ../data/imagenet2012
