#!/bin/bash

# 这个脚本用于运行Python脚本
# 数据集放在同级目录下  ./data

# 确保脚本在遇到错误时停止执行
set -e

export MLU_VISIBLE_DEVICES=0

DATASET_ROOT="/dataset/VOC2007"
[ ! -d "$DATASET_ROOT/VOCdevkit" ] && {
    echo "创建数据集符号链接..."
    mkdir -p $(dirname "$DATASET_ROOT")
    ln -sf /data1/shared/Dataset/VOCdevkit "$DATASET_ROOT/VOCdevkit"
}

# 参数解释
# dataset_path 数据集保存位置
# dataset_path
#   - VOCtrainval_06-Nov-2007.tar
# VOC_year 指定解析数据集的方法
# input_size 输入模型的图像大小
# classes 模型输出类别数量
python eval.py \
       --device mlu \
       --dataset_path "$DATASET_ROOT" \
       --VOC_year 2007 \
       --input_size 256 \
       --num_classes 21 \

# mIOU 0.468
